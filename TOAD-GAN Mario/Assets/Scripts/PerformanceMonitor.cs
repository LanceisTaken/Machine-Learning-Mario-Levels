using System;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.InputSystem;

/// <summary>
/// Tracks and displays real-time performance metrics during gameplay,
/// with a focus on the level-generation pipeline.
///
/// Metrics
/// -------
///  - FPS (current / avg / min over a rolling window)
///  - Frame time (ms) — proxy for per-frame CPU load
///  - Unity allocated memory (MB) and Mono heap usage (MB)
///  - GC collection count (delta per 0.5 s sample)
///  - ML inference time per chunk (last / avg)
///  - Tile instantiation time per chunk (last / avg)
///  - Active tile count in the scene
///  - Total chunks generated this session
///
/// Controls
/// --------
///  Press <see cref="toggleKey"/> (default F3) to show / hide the overlay.
///
/// CSV Logging
/// -----------
///  Enable <see cref="enableCsvLogging"/> to write one row per second to
///  <c>Application.persistentDataPath/perf_log_&lt;timestamp&gt;.csv</c>.
///  Open Unity - Edit - Preferences - Player - "Open Persistent Data Folder"
///  to locate the file after a play session.
///
/// Setup
/// -----
///  1. Attach this component to any persistent GameObject (e.g. GameManager).
///  2. The component auto-discovers <see cref="ToadGanGenerator"/> and
///     <see cref="LevelInstantiator"/> via FindObjectOfType — no Inspector
///     wiring needed unless you want to be explicit.
/// </summary>
public class PerformanceMonitor : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("References (auto-discovered if left empty)")]
    public ToadGanGenerator  generator;
    public LevelInstantiator levelInstantiator;

    [Header("FPS Tracking")]
    [Tooltip("Number of frames kept for rolling average / min calculations.")]
    public int fpsWindowSize = 120;

    [Tooltip("An FPS drop below this value logs a warning to the Console.")]
    public float fpsDropWarningThreshold = 30f;

    [Header("Display")]
    [Tooltip("Key that toggles the on-screen overlay (new Input System).")]
    public Key toggleKey = Key.F3;

    [Tooltip("Show the overlay immediately when the game starts.")]
    public bool showOnStart = true;

    [Header("CSV Logging")]
    [Tooltip("Write performance samples to a CSV file in persistentDataPath.")]
    public bool enableCsvLogging = false;

    [Tooltip("Seconds between CSV rows.")]
    [Range(0.1f, 10f)]
    public float csvLogInterval = 1f;

    // ── HUD style constants ───────────────────────────────────────────────

    private const int   HudWidth      = 295;
    private const int   HudPadding    = 8;
    private const int   HudLineHeight = 18;
    private const float HudAlpha      = 0.82f;

    // ── Runtime state ─────────────────────────────────────────────────────

    private bool _showHud;

    // FPS / frame time — circular buffer instead of Queue to avoid
    // enumerator allocations on every foreach traversal.
    private float[] _fpsRing;
    private int     _fpsRingHead;
    private int     _fpsRingCount;

    private float _currentFps;
    private float _avgFps;
    private float _minFps = float.MaxValue;
    private float _maxFps;
    private float _frameTimeMs;
    private float _dropWarnCooldown;

    // Generation timing (ONNX inference)
    private bool  _isGenerating;
    private float _lastGenMs   = -1f;
    private float _avgGenMs;
    private int   _genCount;
    private float _genTimeSum;

    // Chunk build timing (tile instantiation)
    private float _lastBuildMs   = -1f;
    private float _avgBuildMs;
    private int   _buildCount;
    private float _buildTimeSum;
    private int   _lastChunkTiles;

    // Memory (sampled every 0.5 s to avoid overhead)
    private float _memTimer;
    private const float MemSampleInterval = 0.5f;
    private long  _allocatedBytes;
    private long  _reservedBytes;
    private long  _monoHeapBytes;
    private long  _monoUsedBytes;

    // GC collections (delta per memory sample)
    private int _prevGcCount;
    private int _gcDelta;

    // CSV
    private StreamWriter _csv;
    private float        _csvTimer;

    // Cached reference to the toggle key's ButtonControl — avoids a boxed
    // Enum.IsDefined(typeof(Key), toggleKey) allocation every frame.
    private UnityEngine.InputSystem.Controls.ButtonControl _toggleKeyControl;

    // GUI styles — built once and reused every OnGUI call.
    // Creating new GUIStyle instances inside OnGUI was the single largest
    // source of per-frame managed allocations in the project.
    private GUIStyle _boxStyle;
    private GUIStyle _labelStyle;
    private GUIStyle _headerStyle;
    private GUIStyle _sectionStyle;
    private GUIStyle _rowLabelStyle;
    private bool     _stylesBuilt;

    // ── Unity lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        _showHud = showOnStart;
        _fpsRing = new float[fpsWindowSize];

        if (generator == null)
            generator = FindObjectOfType<ToadGanGenerator>();

        if (levelInstantiator == null)
            levelInstantiator = FindObjectOfType<LevelInstantiator>();

        if (generator != null)
        {
            generator.OnGenerationStarted   += OnGenerationStarted;
            generator.OnGenerationCompleted += OnGenerationCompleted;
        }
        else
        {
            Debug.LogWarning("[PerformanceMonitor] ToadGanGenerator not found — " +
                             "generation timing will not be tracked.");
        }

        if (levelInstantiator != null)
            levelInstantiator.OnChunkBuilt += OnChunkBuilt;
        else
            Debug.LogWarning("[PerformanceMonitor] LevelInstantiator not found — " +
                             "chunk build timing will not be tracked.");

        _prevGcCount = TotalGcCount();

        var kb = Keyboard.current;
        if (kb != null)
        {
            try   { _toggleKeyControl = kb[toggleKey]; }
            catch { _toggleKeyControl = kb[Key.F3]; toggleKey = Key.F3; }
        }

        if (enableCsvLogging)
            OpenCsvLog();
    }

    private void Update()
    {
        if (_toggleKeyControl != null && _toggleKeyControl.wasPressedThisFrame)
            _showHud = !_showHud;

        TrackFps();
        TrackMemory();
        WriteCsvIfDue();
    }

    private void OnDestroy()
    {
        if (generator != null)
        {
            generator.OnGenerationStarted   -= OnGenerationStarted;
            generator.OnGenerationCompleted -= OnGenerationCompleted;
        }

        if (levelInstantiator != null)
            levelInstantiator.OnChunkBuilt -= OnChunkBuilt;

        _csv?.Flush();
        _csv?.Close();
    }

    // ── Event handlers ────────────────────────────────────────────────────

    private void OnGenerationStarted()
    {
        _isGenerating = true;
    }

    private void OnGenerationCompleted(float durationMs)
    {
        _isGenerating  = false;
        _lastGenMs     = durationMs;
        _genCount++;
        _genTimeSum   += durationMs;
        _avgGenMs      = _genTimeSum / _genCount;
    }

    private void OnChunkBuilt(float buildTimeMs, int tileCount)
    {
        _lastBuildMs    = buildTimeMs;
        _lastChunkTiles = tileCount;
        _buildCount++;
        _buildTimeSum  += buildTimeMs;
        _avgBuildMs     = _buildTimeSum / _buildCount;
    }

    // ── Per-frame tracking ────────────────────────────────────────────────

    private void TrackFps()
    {
        float dt = Time.unscaledDeltaTime;
        _frameTimeMs = dt * 1000f;
        _currentFps  = dt > 0f ? 1f / dt : 0f;

        // Write into the circular buffer (no allocation)
        _fpsRing[_fpsRingHead] = _currentFps;
        _fpsRingHead = (_fpsRingHead + 1) % _fpsRing.Length;
        if (_fpsRingCount < _fpsRing.Length)
            _fpsRingCount++;

        // Compute stats with a simple for-loop over the array
        float sum = 0f;
        _minFps = float.MaxValue;
        _maxFps = 0f;

        for (int i = 0; i < _fpsRingCount; i++)
        {
            float f = _fpsRing[i];
            sum += f;
            if (f < _minFps) _minFps = f;
            if (f > _maxFps) _maxFps = f;
        }

        _avgFps = _fpsRingCount > 0 ? sum / _fpsRingCount : 0f;

        _dropWarnCooldown -= dt;
        if (_currentFps < fpsDropWarningThreshold && _dropWarnCooldown <= 0f)
        {
#if UNITY_EDITOR
            Debug.LogWarning($"[PerformanceMonitor] FPS drop detected: " +
                             $"{_currentFps:F1} fps  ({_frameTimeMs:F1} ms/frame)");
#endif
            _dropWarnCooldown = 2f;
        }
    }

    private void TrackMemory()
    {
        _memTimer += Time.unscaledDeltaTime;
        if (_memTimer < MemSampleInterval) return;
        _memTimer = 0f;

        _allocatedBytes = Profiler.GetTotalAllocatedMemoryLong();
        _reservedBytes  = Profiler.GetTotalReservedMemoryLong();
        _monoHeapBytes  = Profiler.GetMonoHeapSizeLong();
        _monoUsedBytes  = Profiler.GetMonoUsedSizeLong();

        int current = TotalGcCount();
        _gcDelta     = current - _prevGcCount;
        _prevGcCount = current;
    }

    private static int TotalGcCount() =>
        GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2);

    // ── OnGUI overlay ─────────────────────────────────────────────────────

    private void OnGUI()
    {
        if (!_showHud) return;

        EnsureStyles();

        int activeTiles = levelInstantiator != null
            ? levelInstantiator.transform.childCount
            : -1;

        int lineCount = 16;
        int totalHeight = HudPadding * 2 + lineCount * HudLineHeight + 4;

        Rect area = new Rect(10, 10, HudWidth, totalHeight);

        Color prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, HudAlpha);
        GUI.Box(area, GUIContent.none, _boxStyle);
        GUI.color = prev;

        GUILayout.BeginArea(new Rect(
            area.x + HudPadding,
            area.y + HudPadding,
            area.width  - HudPadding * 2,
            area.height - HudPadding * 2));

        // ── Header ────────────────────────────────────────────────────────
        DrawHeader($"PERFORMANCE  [{toggleKey}]");

        // ── FPS / Frame time ──────────────────────────────────────────────
        DrawSectionLabel("FRAME RATE");

        string fpsHex = FpsColorHex(_currentFps);
        string ftHex  = FrameTimeColorHex(_frameTimeMs);

        DrawRow("FPS",
            $"<color=#{fpsHex}>{_currentFps:F0}</color>  " +
            $"avg <color=#{fpsHex}>{_avgFps:F0}</color>  " +
            $"min <color=#{FpsColorHex(_minFps)}>{_minFps:F0}</color>");

        DrawRow("Frame Time",
            $"<color=#{ftHex}>{_frameTimeMs:F2} ms</color>");

        // ── Memory ────────────────────────────────────────────────────────
        DrawSectionLabel("MEMORY  (sampled every 0.5 s)");

        DrawRow("Allocated",  $"{BytesToMb(_allocatedBytes):F1} MB");
        DrawRow("Reserved",   $"{BytesToMb(_reservedBytes):F1} MB");
        DrawRow("Mono Heap",  $"{BytesToMb(_monoHeapBytes):F1} MB  " +
                              $"used {BytesToMb(_monoUsedBytes):F1} MB");

        string gcHex = _gcDelta > 0 ? HexYellow : HexGreen;
        DrawRow("GC Allocs",
            $"<color=#{gcHex}>{_gcDelta} collection(s) / 0.5 s</color>");

        // ── Level generation ──────────────────────────────────────────────
        DrawSectionLabel("LEVEL GENERATION");

        string genStatus = _isGenerating
            ? "<color=#FF6B6B>● Running</color>"
            : "<color=#A8FF78>● Idle</color>";
        DrawRow("ML Inference", genStatus);

        DrawRow("Last Gen",  _lastGenMs  >= 0 ? $"{_lastGenMs:F1} ms"  : "—");
        DrawRow("Avg Gen",   _genCount   >  0 ? $"{_avgGenMs:F1} ms  (n={_genCount})"  : "—");
        DrawRow("Last Build",_lastBuildMs>= 0 ? $"{_lastBuildMs:F1} ms" : "—");
        DrawRow("Avg Build", _buildCount >  0 ? $"{_avgBuildMs:F1} ms  (n={_buildCount})" : "—");
        DrawRow("Tiles/Chunk", _lastChunkTiles > 0 ? _lastChunkTiles.ToString() : "—");
        DrawRow("Active Tiles", activeTiles >= 0 ? activeTiles.ToString() : "—");

        GUILayout.EndArea();
    }

    // ── GUI helpers ───────────────────────────────────────────────────────

    private void EnsureStyles()
    {
        if (_stylesBuilt) return;
        _stylesBuilt = true;

        _boxStyle = new GUIStyle(GUI.skin.box)
        {
            normal = { background = Texture2D.whiteTexture }
        };

        _labelStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize  = 12,
            richText  = true,
            alignment = TextAnchor.MiddleLeft,
            normal    = { textColor = Color.white },
            fixedHeight = HudLineHeight
        };

        _headerStyle = new GUIStyle(_labelStyle)
        {
            fontStyle = FontStyle.Bold,
            fontSize  = 13,
            normal    = { textColor = new Color(0.9f, 0.9f, 0.9f) }
        };

        _sectionStyle = new GUIStyle(_labelStyle)
        {
            fontStyle = FontStyle.Bold,
            normal    = { textColor = new Color(0.7f, 0.85f, 1f) }
        };

        _rowLabelStyle = new GUIStyle(_labelStyle)
        {
            normal     = { textColor = new Color(0.75f, 0.75f, 0.75f) },
            fixedWidth = 90
        };
    }

    private void DrawHeader(string text)
    {
        GUILayout.Label(text, _headerStyle);
    }

    private void DrawSectionLabel(string text)
    {
        GUILayout.Label(text, _sectionStyle);
    }

    private void DrawRow(string label, string value)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(label, _rowLabelStyle);
        GUILayout.Label(value, _labelStyle);
        GUILayout.EndHorizontal();
    }

    // ── Color helpers ─────────────────────────────────────────────────────
    // Pre-computed hex strings so OnGUI never calls ColorUtility.ToHtmlStringRGB,
    // which allocates a new managed string on every call.  These are computed
    // exactly once at class load.
    private static readonly string HexGreen  = ColorUtility.ToHtmlStringRGB(new Color(0.4f, 1f,    0.4f));
    private static readonly string HexYellow = ColorUtility.ToHtmlStringRGB(new Color(1f,   0.85f, 0.2f));
    private static readonly string HexRed    = ColorUtility.ToHtmlStringRGB(new Color(1f,   0.35f, 0.35f));
    private static readonly string HexBlue   = ColorUtility.ToHtmlStringRGB(new Color(0.7f, 0.85f, 1f));

    private static string FpsColorHex(float fps)
    {
        if (fps >= 60f) return HexGreen;
        if (fps >= 30f) return HexYellow;
        return HexRed;
    }

    private static string FrameTimeColorHex(float ms)
    {
        if (ms <= 16.67f) return HexGreen;
        if (ms <= 33.33f) return HexYellow;
        return HexRed;
    }

    private static float BytesToMb(long bytes) => bytes / 1_048_576f;

    // ── CSV logging ───────────────────────────────────────────────────────

    private void OpenCsvLog()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string path      = Path.Combine(Application.persistentDataPath,
                                        $"perf_log_{timestamp}.csv");

        try
        {
            _csv = new StreamWriter(path, append: false, encoding: Encoding.UTF8);
            _csv.WriteLine(
                "GameTime_s,FPS,AvgFPS,MinFPS,FrameTime_ms," +
                "Allocated_MB,Reserved_MB,MonoHeap_MB,MonoUsed_MB," +
                "GC_Delta,Generating," +
                "LastGen_ms,AvgGen_ms,GenCount," +
                "LastBuild_ms,AvgBuild_ms,BuildCount," +
                "TilesPerChunk,ActiveTiles");
            _csv.Flush();

            Debug.Log($"[PerformanceMonitor] CSV log opened: {path}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[PerformanceMonitor] Could not open CSV log: {ex.Message}");
            _csv = null;
        }
    }

    private void WriteCsvIfDue()
    {
        if (!enableCsvLogging || _csv == null) return;

        _csvTimer += Time.unscaledDeltaTime;
        if (_csvTimer < csvLogInterval) return;
        _csvTimer = 0f;

        int activeTiles = levelInstantiator != null
            ? levelInstantiator.transform.childCount
            : -1;

        _csv.WriteLine(
            $"{Time.time:F2}," +
            $"{_currentFps:F1},{_avgFps:F1},{_minFps:F1},{_frameTimeMs:F2}," +
            $"{BytesToMb(_allocatedBytes):F2},{BytesToMb(_reservedBytes):F2}," +
            $"{BytesToMb(_monoHeapBytes):F2},{BytesToMb(_monoUsedBytes):F2}," +
            $"{_gcDelta},{(_isGenerating ? 1 : 0)}," +
            $"{_lastGenMs:F2},{_avgGenMs:F2},{_genCount}," +
            $"{_lastBuildMs:F2},{_avgBuildMs:F2},{_buildCount}," +
            $"{_lastChunkTiles},{activeTiles}");

        _csv.Flush();
    }
}
