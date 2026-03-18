using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;
using Unity.InferenceEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

/// <summary>
/// Runs the TOAD-GAN multi-scale generation pipeline locally via Unity Sentis
/// (com.unity.ai.inference) — no Python server required.
///
/// All ML inference and data processing run off the main thread:
///   Phase 1  – Noise tensor data built on a thread-pool thread (System.Random).
///   Phase 2  – Tensor creation, SetInput, Schedule on the main thread (fast GPU dispatch).
///   Phase 3  – Non-blocking async readback polled each frame (ReadbackRequest / IsReadbackRequestDone).
///   Phase 4  – Argmax + post-processing on a thread-pool thread.
///   Phase 5  – Result delivered to subscribers on the main thread.
///
/// Setup
/// -----
/// 1. Attach to any GameObject.
/// 2. Drag the imported toadgan.onnx asset onto <see cref="modelAsset"/> in
///    the Inspector, OR leave it null to auto-load from StreamingAssets.
/// 3. Call <see cref="RequestGeneration"/> (or set <see cref="generateOnStart"/>).
/// 4. Subscribe to <see cref="OnLevelGenerated"/> to receive the tile grid.
/// </summary>
public class ToadGanGenerator : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("Model")]
    [Tooltip("Drag the imported toadgan.onnx asset from the Project window here.")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("Backend")]
    [Tooltip("Preferred Sentis backend. Falls back to CPU if the chosen " +
             "backend is unavailable.")]
    [SerializeField] private BackendType preferredBackend = BackendType.GPUCompute;

    [Header("Generation Parameters")]
    [Tooltip("Noise temperature. > 1 = more variation, < 1 = closer to training distribution.")]
    [SerializeField] [Range(0.1f, 3f)] private float temperature = 1f;

    [Tooltip("Width multiplier relative to the training level size. 1 = original width.")]
    [SerializeField] [Range(0.25f, 4f)] private float scaleW = 1f;

    [Tooltip("Height multiplier relative to the training level size. 1 = original height.")]
    [SerializeField] [Range(0.25f, 4f)] private float scaleH = 1f;

    [Header("Options")]
    [Tooltip("Fire a generation immediately when the component starts. " +
             "Disable when LevelInstantiator controls generation timing.")]
    [SerializeField] private bool generateOnStart = false;

    // ── Events ────────────────────────────────────────────────────────────

    /// <summary>
    /// Raised after a successful generation.  Arguments:
    /// <list type="bullet">
    ///   <item><c>tileIds</c>  – [height][width] argmax tile indices</item>
    ///   <item><c>tileMap</c>  – index-to-character mapping from vocab.json</item>
    ///   <item><c>height</c></item>
    ///   <item><c>width</c></item>
    /// </list>
    /// </summary>
    public event Action<int[][], Dictionary<string, string>, int, int> OnLevelGenerated;

    /// <summary>Raised when something goes wrong during load or inference.</summary>
    public event Action<string> OnError;

    /// <summary>Raised immediately before inference begins.</summary>
    public event Action OnGenerationStarted;

    /// <summary>
    /// Raised after a successful generation with the wall-clock duration of
    /// the full inference pass (noise build + ONNX schedule + readback +
    /// post-processing) in milliseconds.
    /// </summary>
    public event Action<float> OnGenerationCompleted;

    // ── Runtime state ─────────────────────────────────────────────────────

    private Worker _worker;
    private ToadGanMeta _meta;
    private Dictionary<string, string> _tileMap;   // itos: id → char
    private Dictionary<string, int> _charToId;      // stoi: char → id
    private bool _ready;

    // ── Reusable buffers (allocated once, reused across jobs) ─────────────
    // Jobs are sequential (one at a time) so these are never accessed
    // concurrently, even though Phase 1/4 run on background threads.

    /// <summary>Single Stopwatch reused via Restart() — avoids one object per job.</summary>
    private readonly Stopwatch _stopwatch = new Stopwatch();

    /// <summary>Backing array for the temperature scalar tensor — avoids new float[1] per job.</summary>
    private readonly float[] _tempScalarBuf = new float[1];

    /// <summary>
    /// Per-scale noise float[] buffers.  These are the biggest allocation
    /// savings (~1–3 MB total across 5 scales).  Only reallocated when the
    /// required length changes (i.e. scaleH/scaleW differ between jobs).
    /// Written by the background thread in Phase 1, read by the main thread
    /// in Phase 2.  Safe because the coroutine yields until Phase 1 completes,
    /// providing a happens-before guarantee.
    /// </summary>
    private float[][] _noiseBuffers;

    /// <summary>Cached length of each entry in _noiseBuffers for exact-size checks.</summary>
    private int[] _noiseLengths;

    /// <summary>Per-scale [b,c,h,w] shape arrays, reused across jobs.</summary>
    private int[][] _noiseShapes;

    /// <summary>Reused container for Sentis noise tensor references — avoids new Tensor[] per job.</summary>
    private Tensor<float>[] _noiseTensorRefs;

    // ── Async generation infrastructure ───────────────────────────────────

    /// <summary>FIFO queue of generation requests waiting to be processed.</summary>
    private readonly Queue<GenerationJob> _pendingJobs = new Queue<GenerationJob>();

    /// <summary>The job currently being processed (null when idle).</summary>
    private GenerationJob _activeJob;

    /// <summary>Handle to the active processing coroutine for cleanup.</summary>
    private Coroutine _activeCoroutine;

    /// <summary>Number of pending generation requests in the queue.</summary>
    public int PendingJobCount => _pendingJobs.Count;

    /// <summary>True if a generation job is currently being processed.</summary>
    public bool IsProcessing => _activeJob != null;

    /// <summary>
    /// Number of frames the last async readback polled before the GPU result
    /// was ready.  Zero means the GPU finished within one frame.
    /// </summary>
    public int LastAsyncPollFrames { get; private set; }

    /// <summary>
    /// True if the last generation had to fall back to a blocking readback
    /// because the async API was unavailable.
    /// </summary>
    public bool LastJobBlockedMainThread { get; private set; }

    /// <summary>
    /// Cumulative count of generations that required a blocking main-thread
    /// readback (ideally stays at zero).
    /// </summary>
    public int TotalBlockedCount { get; private set; }

    /// <summary>Total number of successfully completed generations.</summary>
    public int TotalGenerations { get; private set; }

    // ── Job data class ────────────────────────────────────────────────────

    public class GenerationJob
    {
        public int   ChunkIndex;
        public int   Seed;   // -1 = use a random seed
        public float Temperature;
        public float ScaleW;
        public float ScaleH;
    }

    // ── Unity lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        try
        {
            LoadMetaAndVocab();
            LoadModel();
            InitReusableBuffers();
            _ready = true;
        }
        catch (Exception ex)
        {
            string msg = $"[ToadGanGenerator] Initialisation failed: {ex.Message}";
            Debug.LogError(msg);
            OnError?.Invoke(msg);
            return;
        }

        if (generateOnStart)
            RequestGeneration();
    }

    private void Update()
    {
        if (_activeJob == null && _pendingJobs.Count > 0)
        {
            _activeJob = _pendingJobs.Dequeue();
            _activeCoroutine = StartCoroutine(ProcessJobCoroutine(_activeJob));
        }
    }

    private void OnDestroy()
    {
        if (_activeCoroutine != null)
            StopCoroutine(_activeCoroutine);
        DisposeNoiseTensors();
        _worker?.Dispose();
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Enqueue a non-blocking generation request.  The result arrives via
    /// <see cref="OnLevelGenerated"/>.  Multiple requests are processed in
    /// FIFO order, one at a time.
    /// </summary>
    /// <param name="chunkIndex">Optional chunk index for tracking.</param>
    /// <param name="seed">RNG seed for reproducibility; -1 = random.</param>
    public void RequestGeneration(int chunkIndex = -1, int seed = -1)
    {
        if (!_ready)
        {
            string msg = "[ToadGanGenerator] Not initialised — cannot generate.";
            Debug.LogError(msg);
            OnError?.Invoke(msg);
            return;
        }

        _pendingJobs.Enqueue(new GenerationJob
        {
            ChunkIndex  = chunkIndex,
            Seed        = seed,
            Temperature = temperature,
            ScaleW      = scaleW,
            ScaleH      = scaleH
        });
    }

    /// <summary>
    /// Backward-compatible entry point.  Calls <see cref="RequestGeneration"/>
    /// with default parameters so existing callers keep working.
    /// </summary>
    public void Generate() => RequestGeneration();

    // ── Async processing coroutine ────────────────────────────────────────

    private IEnumerator ProcessJobCoroutine(GenerationJob job)
    {
        OnGenerationStarted?.Invoke();
        _stopwatch.Restart();
        int  pollFrames = 0;
        bool didBlock   = false;

        Tensor<float>   temperatureTensor = null;
        Tensor<float>   outputRef        = null;
        bool            asyncOk          = true;

        // ── Phase 1: Build noise data on a thread-pool thread ────
        // Reuses _noiseBuffers / _noiseShapes — only reallocates when
        // the required size changes.  Safe because jobs are sequential.
        var noiseTask = Task.Run(() =>
        {
            var rng = job.Seed >= 0
                ? new System.Random(job.Seed)
                : new System.Random();
            BuildNoiseDataThreadSafe(rng, job.ScaleH, job.ScaleW);
        });

        while (!noiseTask.IsCompleted)
            yield return null;

        try
        {
            if (noiseTask.IsFaulted)
                throw noiseTask.Exception?.InnerException ?? noiseTask.Exception;

            // ── Phase 2: Create tensors & schedule (main thread, fast) ──
            // Dispose the PREVIOUS job's noise tensors right before creating
            // new ones.  This avoids invalidating the Worker's internal input
            // bindings between Schedule() and readback completion — the Worker
            // never sees a disposed tensor while it still has pending work.
            DisposeNoiseTensors();

            _tempScalarBuf[0] = job.Temperature;
            temperatureTensor = new Tensor<float>(new TensorShape(), _tempScalarBuf);
            _worker.SetInput("temperature", temperatureTensor);

            // Reuse _noiseTensorRefs container — avoids new Tensor<float>[] each job.
            for (int s = 0; s < _meta.num_scales; s++)
            {
                int[] ns = _noiseShapes[s];
                var shape = new TensorShape(ns[0], ns[1], ns[2], ns[3]);
                _noiseTensorRefs[s] = new Tensor<float>(shape, _noiseBuffers[s]);
                _worker.SetInput(_meta.input_names[s + 1], _noiseTensorRefs[s]);
            }

            _worker.Schedule();

            outputRef = _worker.PeekOutput(_meta.output_name) as Tensor<float>;
            try
            {
                outputRef.ReadbackRequest();
            }
            catch
            {
                asyncOk = false;
            }
        }
        catch (Exception ex)
        {
            _stopwatch.Stop();
            Debug.LogError($"[ToadGanGenerator] Inference failed: {ex.Message}");
            OnError?.Invoke(ex.Message);
            _activeJob       = null;
            _activeCoroutine = null;
            temperatureTensor?.Dispose();
            DisposeNoiseTensors();
            yield break;
        }

        // ── Phase 3: Non-blocking async readback (yield outside try-catch) ──
        if (asyncOk)
        {
            while (!outputRef.IsReadbackRequestDone())
            {
                pollFrames++;
                yield return null;
            }
        }
        else
        {
            yield return null;
            didBlock = true;
        }

        int channels = outputRef.shape[1];
        int height   = outputRef.shape[2];
        int width    = outputRef.shape[3];

        float[] rawOutput = null;
        try
        {
            rawOutput = outputRef.DownloadToArray();
        }
        catch (Exception ex)
        {
            _stopwatch.Stop();
            Debug.LogError($"[ToadGanGenerator] Readback failed: {ex.Message}");
            OnError?.Invoke(ex.Message);
            _activeJob       = null;
            _activeCoroutine = null;
            temperatureTensor?.Dispose();
            DisposeNoiseTensors();
            yield break;
        }

        // Dispose the temperature tensor now (small, not reused).
        // Noise tensors are NOT disposed here — they stay alive so the
        // Worker's internal bindings remain valid.  They are disposed at
        // the start of the NEXT job's Phase 2 (or in OnDestroy).
        temperatureTensor?.Dispose();
        temperatureTensor = null;

        // ── Phase 4: Argmax + post-processing on background thread ──
        int[][] tileIds  = null;
        var     charToId = _charToId;

        var postTask = Task.Run(() =>
        {
            tileIds = ArgmaxChannelsFromArray(
                rawOutput, channels, height, width);
            FixPipesStatic(tileIds, height, width, charToId);
            FixBlocksOnPipesStatic(tileIds, height, width, charToId);
            FixLuckyBlocksStatic(tileIds, height, width, charToId);
        });

        while (!postTask.IsCompleted)
            yield return null;

        try
        {
            if (postTask.IsFaulted)
                throw postTask.Exception?.InnerException ?? postTask.Exception;

            _stopwatch.Stop();
            float durationMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

            LastAsyncPollFrames      = pollFrames;
            LastJobBlockedMainThread = didBlock;
            if (didBlock) TotalBlockedCount++;
            TotalGenerations++;

            Debug.Log($"[ToadGanGenerator] Generated {height}×{width} tile grid " +
                      $"({channels} channels) in {durationMs:F1} ms " +
                      $"[async: {pollFrames} poll frames, blocked={didBlock}]");

            OnLevelGenerated?.Invoke(tileIds, _tileMap, height, width);
            OnGenerationCompleted?.Invoke(durationMs);
        }
        catch (Exception ex)
        {
            _stopwatch.Stop();
            Debug.LogError($"[ToadGanGenerator] Post-process failed: {ex.Message}");
            OnError?.Invoke(ex.Message);
        }
        finally
        {
            _activeJob       = null;
            _activeCoroutine = null;
        }
    }

    // ── Initialisation helpers ────────────────────────────────────────────

    private void LoadMetaAndVocab()
    {
        string toadganDir = Path.Combine(Application.streamingAssetsPath, "TOADGAN");

        string metaJson = File.ReadAllText(Path.Combine(toadganDir, "toadgan_meta.json"));
        _meta = JsonConvert.DeserializeObject<ToadGanMeta>(metaJson);

        string vocabJson = File.ReadAllText(Path.Combine(toadganDir, "vocab.json"));
        var vocab = JsonConvert.DeserializeObject<VocabFile>(vocabJson);
        _tileMap = vocab.itos;

        _charToId = new Dictionary<string, int>();
        foreach (var kvp in vocab.stoi)
        {
            if (int.TryParse(kvp.Value, out int id))
                _charToId[kvp.Key] = id;
        }

        Debug.Log($"[ToadGanGenerator] Meta loaded: {_meta.num_scales} scales, " +
                  $"{_meta.num_tokens} tokens.");
    }

    private void LoadModel()
    {
        if (modelAsset == null)
            throw new InvalidOperationException(
                "Model Asset is not assigned! Drag the toadgan.onnx asset " +
                "from Assets/StreamingAssets/TOADGAN/ into the Model Asset " +
                "field in the Inspector.");

        Model model = ModelLoader.Load(modelAsset);

        try
        {
            _worker = new Worker(model, preferredBackend);
            Debug.Log($"[ToadGanGenerator] Worker created ({preferredBackend}).");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ToadGanGenerator] {preferredBackend} unavailable " +
                             $"({ex.Message}). Falling back to CPU.");
            _worker = new Worker(model, BackendType.CPU);
        }
    }

    // ── Reusable buffer management ──────────────────────────────────────

    /// <summary>Allocate the container arrays sized for num_scales.</summary>
    private void InitReusableBuffers()
    {
        int n = _meta.num_scales;
        _noiseBuffers    = new float[n][];
        _noiseLengths    = new int[n];
        _noiseShapes     = new int[n][];
        _noiseTensorRefs = new Tensor<float>[n];

        for (int s = 0; s < n; s++)
            _noiseShapes[s] = new int[4];
    }

    /// <summary>Dispose any live tensors in the reusable array and null the slots.</summary>
    private void DisposeNoiseTensors()
    {
        if (_noiseTensorRefs == null) return;
        for (int s = 0; s < _noiseTensorRefs.Length; s++)
        {
            _noiseTensorRefs[s]?.Dispose();
            _noiseTensorRefs[s] = null;
        }
    }

    // ── Thread-safe noise generation (reuses _noiseBuffers) ──────────────
    // Uses System.Random + System.Math so it can run on any thread.
    // The reusable _noiseBuffers / _noiseLengths / _noiseShapes arrays are
    // safe to mutate here because jobs run one at a time (sequential queue).

    private void BuildNoiseDataThreadSafe(
        System.Random rng, float scaleHeight, float scaleWidth)
    {
        for (int s = 0; s < _meta.num_scales; s++)
        {
            int[] baseShape = _meta.pyramid_shapes[s];
            int b = baseShape[0];
            int c = baseShape[1];
            int h = Math.Max(1, (int)Math.Round(baseShape[2] * (double)scaleHeight));
            int w = Math.Max(1, (int)Math.Round(baseShape[3] * (double)scaleWidth));

            _noiseShapes[s][0] = b;
            _noiseShapes[s][1] = c;
            _noiseShapes[s][2] = h;
            _noiseShapes[s][3] = w;

            int length = b * c * h * w;

            // Reallocate when the required length differs from the cached buffer.
            // Using != (not <) ensures Sentis always receives an exactly-sized
            // array, avoiding wasted GPU upload bandwidth from oversized buffers.
            if (_noiseBuffers[s] == null || _noiseLengths[s] != length)
            {
                _noiseBuffers[s] = new float[length];
                _noiseLengths[s] = length;
            }

            FillGaussianNoiseThreadSafe(rng, _noiseBuffers[s], length);
        }
    }

    /// <summary>
    /// Box-Muller in-place fill using System.Random (thread-safe).
    /// Writes into an existing buffer — zero allocation.
    /// </summary>
    private static void FillGaussianNoiseThreadSafe(System.Random rng, float[] buf, int count)
    {
        for (int i = 0; i < count; i += 2)
        {
            double u1 = rng.NextDouble();
            if (u1 < 1e-10) u1 = 1e-10;
            double u2 = rng.NextDouble();
            double r  = Math.Sqrt(-2.0 * Math.Log(u1));
            double th = 2.0 * Math.PI * u2;

            buf[i] = (float)(r * Math.Cos(th));
            if (i + 1 < count)
                buf[i + 1] = (float)(r * Math.Sin(th));
        }
    }

    // ── Thread-safe post-processing (static — no instance state) ─────────

    /// <summary>
    /// Argmax across the channel axis of a flat NCHW float array,
    /// producing a [H][W] grid of tile indices.
    /// </summary>
    private static int[][] ArgmaxChannelsFromArray(
        float[] data, int channels, int height, int width)
    {
        var grid = new int[height][];

        for (int row = 0; row < height; row++)
        {
            grid[row] = new int[width];
            for (int col = 0; col < width; col++)
            {
                int   bestIdx = 0;
                float bestVal = data[row * width + col];   // channel 0

                for (int c = 1; c < channels; c++)
                {
                    float v = data[(c * height + row) * width + col];
                    if (v > bestVal)
                    {
                        bestVal = v;
                        bestIdx = c;
                    }
                }

                grid[row][col] = bestIdx;
            }
        }

        return grid;
    }

    private static int TryGetIdStatic(Dictionary<string, int> charToId, string ch) =>
        charToId.TryGetValue(ch, out int id) ? id : -1;

    private static void FixPipesStatic(
        int[][] grid, int rows, int cols,
        Dictionary<string, int> charToId, int minHeight = 3)
    {
        int PIPE   = TryGetIdStatic(charToId, "§");
        int GROUND = TryGetIdStatic(charToId, "#");
        int SKY    = TryGetIdStatic(charToId, "-");
        if (PIPE < 0 || GROUND < 0 || SKY < 0) return;

        for (int c = 0; c < cols; c++)
        {
            int groundRow = -1;
            for (int r = rows - 1; r >= 0; r--)
            {
                if (grid[r][c] == GROUND) { groundRow = r; break; }
            }

            int topPipe = -1;
            for (int r = 0; r < rows; r++)
            {
                if (grid[r][c] == PIPE) { topPipe = r; break; }
            }

            if (topPipe < 0) continue;

            if (groundRow < 0 || topPipe >= groundRow)
            {
                for (int r = 0; r < rows; r++)
                    if (grid[r][c] == PIPE) grid[r][c] = SKY;
                continue;
            }

            for (int r = topPipe; r < groundRow; r++)
                grid[r][c] = PIPE;

            int pipeHeight = groundRow - topPipe;
            if (pipeHeight < minHeight)
            {
                int newTop = groundRow - minHeight;
                if (newTop < 0)
                {
                    for (int r = 0; r < rows; r++)
                        if (grid[r][c] == PIPE) grid[r][c] = SKY;
                    continue;
                }

                bool canExtend = true;
                for (int r = newTop; r < topPipe; r++)
                {
                    if (grid[r][c] != SKY) { canExtend = false; break; }
                }

                if (canExtend)
                {
                    for (int r = newTop; r < topPipe; r++)
                        grid[r][c] = PIPE;
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                        if (grid[r][c] == PIPE) grid[r][c] = SKY;
                }
            }
        }
    }

    private static readonly string[] _pipeCharKeys = { "§", "<", ">", "[", "]" };

    private static void FixBlocksOnPipesStatic(
        int[][] grid, int rows, int cols,
        Dictionary<string, int> charToId, int clearance = 2)
    {
        int SKY = TryGetIdStatic(charToId, "-");
        if (SKY < 0) return;

        var pipeIds = new HashSet<int>();
        foreach (string ch in _pipeCharKeys)
        {
            int id = TryGetIdStatic(charToId, ch);
            if (id >= 0) pipeIds.Add(id);
        }
        if (pipeIds.Count == 0) return;

        for (int c = 0; c < cols; c++)
        {
            int topPipe = -1;
            for (int r = 0; r < rows; r++)
            {
                if (pipeIds.Contains(grid[r][c])) { topPipe = r; break; }
            }
            if (topPipe < 0) continue;

            for (int offset = 1; offset <= clearance; offset++)
            {
                int above = topPipe - offset;
                if (above >= 0 && grid[above][c] != SKY)
                    grid[above][c] = SKY;
            }
        }
    }

    private static void FixLuckyBlocksStatic(
        int[][] grid, int rows, int cols,
        Dictionary<string, int> charToId, int minGap = 3, int minGapAbove = 1)
    {
        int LUCKY = TryGetIdStatic(charToId, "?");
        int SKY   = TryGetIdStatic(charToId, "-");
        if (LUCKY < 0 || SKY < 0) return;

        var luckyRows = new List<int>();

        for (int c = 0; c < cols; c++)
        {
            for (int r = rows - 1; r >= 0; r--)
            {
                if (grid[r][c] != LUCKY) continue;

                int directlyBelow = r + 1;
                if (directlyBelow >= rows || grid[directlyBelow][c] != SKY)
                {
                    grid[r][c] = SKY;
                    continue;
                }

                int gap = 0;
                for (int below = r + 1; below < rows; below++)
                {
                    if (grid[below][c] == SKY) gap++;
                    else break;
                }
                if (gap < minGap)
                    grid[r][c] = SKY;
            }

            for (int r = 0; r < rows; r++)
            {
                if (grid[r][c] != LUCKY) continue;
                int gapAbove = 0;
                bool hitSolid = false;
                for (int above = r - 1; above >= 0; above--)
                {
                    if (grid[above][c] == SKY) gapAbove++;
                    else { hitSolid = true; break; }
                }
                if (hitSolid && gapAbove < minGapAbove)
                    grid[r][c] = SKY;
            }

            luckyRows.Clear();
            for (int r = rows - 1; r >= 0; r--)
            {
                if (grid[r][c] == LUCKY) luckyRows.Add(r);
            }
            if (luckyRows.Count < 2) continue;

            int lastKept = luckyRows[0];
            for (int i = 1; i < luckyRows.Count; i++)
            {
                int r = luckyRows[i];
                int between = lastKept - r - 1;
                if (between >= minGap)
                    lastKept = r;
                else
                    grid[r][c] = SKY;
            }
        }
    }

    // ── JSON data classes ─────────────────────────────────────────────────

    [Serializable]
    private class ToadGanMeta
    {
        public int      num_scales;
        public int      num_tokens;
        public float[]  noise_amps;
        public int[][]  pyramid_shapes;
        public string[] input_names;
        public string   output_name;
    }

    [Serializable]
    private class VocabFile
    {
        public Dictionary<string, string> stoi;
        public Dictionary<string, string> itos;
    }
}
