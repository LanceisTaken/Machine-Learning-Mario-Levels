using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;
using Unity.InferenceEngine;

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

    // ── Runtime state ─────────────────────────────────────────────────────

    private Worker _worker;
    private ToadGanMeta _meta;
    private Dictionary<string, string> _tileMap;
    private bool _ready;

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

        Tensor<float>[] noiseTensors = null;
        Tensor<float> cpuOutput = null;

        try
        {
            noiseTensors = BuildNoiseTensors();

            for (int s = 0; s < _meta.num_scales; s++)
                _worker.SetInput(_meta.input_names[s], noiseTensors[s]);

            _worker.Schedule();

            var outputRef = _worker.PeekOutput(_meta.output_name) as Tensor<float>;
            cpuOutput = outputRef.ReadbackAndClone();

            int channels = cpuOutput.shape[1];
            int height   = cpuOutput.shape[2];
            int width    = cpuOutput.shape[3];

            int[][] tileIds = ArgmaxChannels(cpuOutput, channels, height, width);

            Debug.Log($"[ToadGanGenerator] Generated {height}×{width} tile grid " +
                      $"({channels} token channels).");

            OnLevelGenerated?.Invoke(tileIds, _tileMap, height, width);
        }
        catch (Exception ex)
        {
            string msg = $"[ToadGanGenerator] Inference failed: {ex.Message}";
            Debug.LogError(msg);
            OnError?.Invoke(msg);
        }
        finally
        {
            if (noiseTensors != null)
                foreach (var t in noiseTensors)
                    t?.Dispose();

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

    private Tensor<float>[] BuildNoiseTensors()
    {
        var tensors = new Tensor<float>[_meta.num_scales];

        for (int s = 0; s < _meta.num_scales; s++)
        {
            int[] shape = _meta.pyramid_shapes[s];
            var tensorShape = new TensorShape(shape[0], shape[1], shape[2], shape[3]);
            int length = shape[0] * shape[1] * shape[2] * shape[3];

            float[] data = GaussianNoise(length);
            tensors[s] = new Tensor<float>(tensorShape, data);
        }

        return tensors;
    }

    /// <summary>Box-Muller transform to sample from N(0,1).</summary>
    private static float[] GaussianNoise(int count)
    {
        var buf = new float[count];
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
        return buf;
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
