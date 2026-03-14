using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;
using Unity.InferenceEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

/// <summary>
/// Runs the TOAD-GAN multi-scale generation pipeline locally via Unity Sentis
/// (com.unity.ai.inference) — no Python server required.
///
/// Setup
/// -----
/// 1. Attach to any GameObject.
/// 2. Drag the imported toadgan.onnx asset onto <see cref="modelAsset"/> in
///    the Inspector, OR leave it null to auto-load from StreamingAssets.
/// 3. Call <see cref="Generate"/> (or set <see cref="generateOnStart"/>).
/// 4. Subscribe to <see cref="OnLevelGenerated"/> to receive the tile grid.
///
/// The component reads pyramid shapes and vocab from the JSON sidecar files
/// that <c>export_onnx.py</c> placed alongside the ONNX model in
/// <c>StreamingAssets/TOADGAN/</c>.
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

    // Reusable collections for post-processing passes.  Allocated once and
    // cleared before each use so chunk generation does not produce GC garbage.
    private readonly HashSet<int> _pipeIds   = new HashSet<int>();
    private readonly List<int>    _luckyRows = new List<int>();

    // ── Reusable generation buffers ──────────────────────────────────────
    // These fields are allocated once (or resized only when generation
    // parameters change) and reused across Generate() calls.  The noise
    // backing arrays are the largest — potentially several MB total for a
    // multi-scale GAN — so recycling them avoids per-chunk GC pressure that
    // would otherwise trigger periodic frame-rate hitches.

    /// <summary>Reused across calls — avoids a Stopwatch heap allocation per Generate().</summary>
    private readonly Stopwatch _stopwatch = new Stopwatch();

    /// <summary>
    /// Single-element backing array for the temperature scalar tensor.
    /// Reused to avoid a trivial but unnecessary per-call array allocation.
    /// </summary>
    private readonly float[] _tempScalarBuf = new float[1];

    /// <summary>
    /// Per-scale noise float[] buffers.  Each buffer is filled with fresh
    /// Box-Muller random values every Generate() call but is only
    /// reallocated when the required element count changes (i.e. scaleH or
    /// scaleW was modified between calls).  For a typical 5-scale model
    /// these buffers total ~1-3 MB — reusing them eliminates the single
    /// largest source of managed allocations in the generation pipeline.
    /// </summary>
    private float[][] _noiseBuffers;

    /// <summary>Current element count of each entry in <see cref="_noiseBuffers"/>.</summary>
    private int[] _noiseLengths;

    /// <summary>
    /// Reusable container for the per-scale Tensor objects passed to the
    /// worker.  The array itself is cached; its elements are overwritten
    /// each call and disposed after inference completes.
    /// </summary>
    private Tensor<float>[] _noiseTensorRefs;

    // ── Unity lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        try
        {
            LoadMetaAndVocab();
            LoadModel();
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
            Generate();
    }

    private void OnDestroy()
    {
        DisposeNoiseTensors();
        _worker?.Dispose();
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Run one full multi-scale generation pass and raise
    /// <see cref="OnLevelGenerated"/> with the resulting tile grid.
    /// </summary>
    public void Generate()
    {
        if (!_ready)
        {
            string msg = "[ToadGanGenerator] Not initialised — cannot generate.";
            Debug.LogError(msg);
            OnError?.Invoke(msg);
            return;
        }

        OnGenerationStarted?.Invoke();
        _stopwatch.Restart();

        Tensor<float> temperatureTensor = null;
        Tensor<float> cpuOutput = null;

        try
        {
            // Temperature scalar — reuses _tempScalarBuf (no array allocation).
            _tempScalarBuf[0] = temperature;
            temperatureTensor = new Tensor<float>(new TensorShape(), _tempScalarBuf);
            _worker.SetInput("temperature", temperatureTensor);

            // Per-scale noise tensors — backing float[] arrays are reused via
            // _noiseBuffers; only the lightweight Tensor wrappers are new.
            BuildNoiseTensors(scaleH, scaleW);

            for (int s = 0; s < _meta.num_scales; s++)
                _worker.SetInput(_meta.input_names[s + 1], _noiseTensorRefs[s]);

            _worker.Schedule();

            var outputRef = _worker.PeekOutput(_meta.output_name) as Tensor<float>;
            cpuOutput = outputRef.ReadbackAndClone();

            int channels = cpuOutput.shape[1];
            int height   = cpuOutput.shape[2];
            int width    = cpuOutput.shape[3];

            // ArgmaxChannels must allocate a fresh int[][] each call because
            // the consumer (LevelInstantiator) holds the reference across
            // multiple frames in a coroutine.
            int[][] tileIds = ArgmaxChannels(cpuOutput, channels, height, width);

            FixPipes(tileIds, height, width);
            FixBlocksOnPipes(tileIds, height, width);
            FixLuckyBlocks(tileIds, height, width);

            _stopwatch.Stop();
            float durationMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

            Debug.Log($"[ToadGanGenerator] Generated {height}×{width} tile grid " +
                      $"({channels} token channels) in {durationMs:F1} ms.");

            OnLevelGenerated?.Invoke(tileIds, _tileMap, height, width);
            OnGenerationCompleted?.Invoke(durationMs);
        }
        catch (Exception ex)
        {
            _stopwatch.Stop();
            string msg = $"[ToadGanGenerator] Inference failed: {ex.Message}";
            Debug.LogError(msg);
            OnError?.Invoke(msg);
        }
        finally
        {
            temperatureTensor?.Dispose();
            DisposeNoiseTensors();
            cpuOutput?.Dispose();
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

    // ── Noise generation ──────────────────────────────────────────────────

    /// <summary>
    /// Populates <see cref="_noiseTensorRefs"/> with one noise tensor per
    /// scale.  The backing <c>float[]</c> arrays in <see cref="_noiseBuffers"/>
    /// are reused across calls — only reallocated when the required element
    /// count changes (e.g. after a scaleH/scaleW tweak in the Inspector).
    /// The <see cref="Tensor{T}"/> wrappers must be freshly constructed each
    /// call because Sentis copies the data on creation and the Worker holds a
    /// native reference until the next Schedule.
    /// </summary>
    private void BuildNoiseTensors(float scaleHeight, float scaleWidth)
    {
        int numScales = _meta.num_scales;

        if (_noiseBuffers == null || _noiseBuffers.Length != numScales)
        {
            _noiseBuffers    = new float[numScales][];
            _noiseLengths    = new int[numScales];
            _noiseTensorRefs = new Tensor<float>[numScales];
        }

        for (int s = 0; s < numScales; s++)
        {
            int[] baseShape = _meta.pyramid_shapes[s];

            int b = baseShape[0];
            int c = baseShape[1];
            int h = Mathf.Max(1, Mathf.RoundToInt(baseShape[2] * scaleHeight));
            int w = Mathf.Max(1, Mathf.RoundToInt(baseShape[3] * scaleWidth));

            var tensorShape = new TensorShape(b, c, h, w);
            int length = b * c * h * w;

            // Resize only when the required element count has changed.
            if (_noiseBuffers[s] == null || _noiseLengths[s] != length)
            {
                _noiseBuffers[s] = new float[length];
                _noiseLengths[s] = length;
            }

            FillGaussianNoise(_noiseBuffers[s], length);
            _noiseTensorRefs[s] = new Tensor<float>(tensorShape, _noiseBuffers[s]);
        }
    }

    /// <summary>
    /// Dispose every tensor in <see cref="_noiseTensorRefs"/> and null
    /// the slots so stale references are never reused.
    /// </summary>
    private void DisposeNoiseTensors()
    {
        if (_noiseTensorRefs == null) return;
        for (int i = 0; i < _noiseTensorRefs.Length; i++)
        {
            _noiseTensorRefs[i]?.Dispose();
            _noiseTensorRefs[i] = null;
        }
    }

    /// <summary>
    /// Box-Muller transform — fills <paramref name="buf"/> in-place with
    /// N(0,1) samples.  No heap allocation.
    /// </summary>
    private static void FillGaussianNoise(float[] buf, int count)
    {
        for (int i = 0; i < count; i += 2)
        {
            float u1 = UnityEngine.Random.Range(float.Epsilon, 1f);
            float u2 = UnityEngine.Random.Range(0f, 1f);
            float r  = Mathf.Sqrt(-2f * Mathf.Log(u1));
            float th = 2f * Mathf.PI * u2;

            buf[i] = r * Mathf.Cos(th);
            if (i + 1 < count)
                buf[i + 1] = r * Mathf.Sin(th);
        }
    }

    // ── Post-processing ───────────────────────────────────────────────────

    /// <summary>
    /// Argmax across the channel (dim-1) axis of a [1, C, H, W] tensor,
    /// producing a [H][W] grid of tile indices.
    /// </summary>
    private static int[][] ArgmaxChannels(
        Tensor<float> tensor, int channels, int height, int width)
    {
        var grid = new int[height][];

        for (int row = 0; row < height; row++)
        {
            grid[row] = new int[width];
            for (int col = 0; col < width; col++)
            {
                int   bestIdx = 0;
                float bestVal = tensor[0, 0, row, col];

                for (int c = 1; c < channels; c++)
                {
                    float v = tensor[0, c, row, col];
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

    // ── Level fix-up passes (ported from generate.py) ──────────────────────

    private int TryGetId(string ch) =>
        _charToId.TryGetValue(ch, out int id) ? id : -1;

    /// <summary>
    /// Ensure pipe columns (§) form unbroken vertical segments touching
    /// ground (#).  Pipes without ground below or that can't reach minimum
    /// height are erased.
    /// </summary>
    private void FixPipes(int[][] grid, int rows, int cols, int minHeight = 3)
    {
        int PIPE   = TryGetId("§");
        int GROUND = TryGetId("#");
        int SKY    = TryGetId("-");
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

    /// <summary>
    /// Clear a buffer of rows above each pipe head so blocks can't sit
    /// directly on top of pipes.
    /// Recognises all pipe tile characters (§, &lt;, &gt;, [, ]) so that the
    /// clearance is applied regardless of which pipe token the vocab uses.
    /// </summary>
    private void FixBlocksOnPipes(int[][] grid, int rows, int cols, int clearance = 2)
    {
        int SKY = TryGetId("-");
        if (SKY < 0) return;

        // Collect every pipe-related tile ID that is present in this vocab.
        // Using a set means the column scan below works even when the model
        // outputs the 4-quadrant tokens (<, >, [, ]) instead of the compound §.
        _pipeIds.Clear();
        foreach (string ch in _pipeCharKeys)
        {
            int id = TryGetId(ch);
            if (id >= 0) _pipeIds.Add(id);
        }
        if (_pipeIds.Count == 0) return;

        for (int c = 0; c < cols; c++)
        {
            int topPipe = -1;
            for (int r = 0; r < rows; r++)
            {
                if (_pipeIds.Contains(grid[r][c])) { topPipe = r; break; }
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

    /// <summary>
    /// Ensure every ? block has enough vertical clearance below (and above)
    /// to be reachable, and enforce minimum spacing between stacked ? blocks.
    /// </summary>
    private void FixLuckyBlocks(int[][] grid, int rows, int cols,
                                int minGap = 3, int minGapAbove = 1)
    {
        int LUCKY = TryGetId("?");
        int SKY   = TryGetId("-");
        if (LUCKY < 0 || SKY < 0) return;

        for (int c = 0; c < cols; c++)
        {
            // Pass 1a: remove ? with insufficient gap below.
            // Two-tier check:
            //   (i)  Hard rule – the tile directly below must always be empty sky,
            //        regardless of minGap.  A lucky block must never sit flush on
            //        any solid surface (ground, pipe top, brick, etc.).
            //   (ii) Soft rule – require minGap consecutive sky tiles below so the
            //        block is reachable and visually readable.
            for (int r = rows - 1; r >= 0; r--)
            {
                if (grid[r][c] != LUCKY) continue;

                // (i) Hard check: tile directly below must be sky.
                int directlyBelow = r + 1;
                if (directlyBelow >= rows || grid[directlyBelow][c] != SKY)
                {
                    grid[r][c] = SKY;
                    continue;
                }

                // (ii) Soft check: count consecutive sky tiles below.
                int gap = 0;
                for (int below = r + 1; below < rows; below++)
                {
                    if (grid[below][c] == SKY) gap++;
                    else break;
                }
                if (gap < minGap)
                    grid[r][c] = SKY;
            }

            // Pass 1b: remove ? with insufficient gap above
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

            // Pass 2: enforce spacing between surviving ? blocks (bottom-up)
            _luckyRows.Clear();
            for (int r = rows - 1; r >= 0; r--)
            {
                if (grid[r][c] == LUCKY) _luckyRows.Add(r);
            }
            if (_luckyRows.Count < 2) continue;

            int lastKept = _luckyRows[0];
            for (int i = 1; i < _luckyRows.Count; i++)
            {
                int r = _luckyRows[i];
                int between = lastKept - r - 1;
                if (between >= minGap)
                    lastKept = r;
                else
                    grid[r][c] = SKY;
            }
        }
    }

    // Allocated once — reused in every FixBlocksOnPipes call.
    private static readonly string[] _pipeCharKeys = { "§", "<", ">", "[", "]" };

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
