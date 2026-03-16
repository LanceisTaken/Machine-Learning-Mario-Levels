using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Cost-aware generation scheduler that sits between <see cref="ToadGanGenerator"/>
/// and <see cref="LevelInstantiator"/>, managing when ML inference runs relative to
/// the player's position and speed.
///
/// Why this exists
/// ───────────────
/// TOAD-GAN inference is expensive (~100–400 ms on GPU, more on CPU).  Without
/// scheduling, the game either stalls the main thread at the last moment (reactive
/// generation) or wastes resources generating far more than needed.
///
/// This scheduler mitigates that cost in four ways:
///
///   1. **Predictive prefetch** – uses an exponential moving average of measured
///      inference + build time combined with the assumed worst-case player speed
///      to compute how many chunks must exist ahead of the player *before* a new
///      generation is triggered.  Formula:
///
///        safePrefetch = ⌈(avgInference + avgBuild) / 1000 × maxSpeed / chunkWidth⌉
///                       + safetyMargin
///
///      This ensures the ML cost is paid while the player still has runway, so the
///      synchronous stall occurs during gameplay that is visually unaffected.
///
///   2. **Startup pre-generation** – front-loads N chunks into a buffer during
///      <c>Start()</c>, shifting the ML cost to the loading phase where a brief
///      delay is expected.
///
///   3. **Chunk buffer** – generated tile data is queued before being handed to
///      <see cref="LevelInstantiator"/> for building.  The instantiator builds
///      one chunk at a time (spread across frames via its coroutine), while the
///      scheduler keeps requesting more to refill the queue.
///
///   4. **Graceful degradation** – if the player out-runs the buffer (e.g. dash
///      sprint), a flat emergency "fallback chunk" is spawned procedurally in
///      microseconds instead of stalling the thread.  This gives the player safe
///      ground to stand on while the next ML chunk catches up.
///
/// Setup
/// ─────
/// 1. Create a config asset: Assets → Create → TOAD-GAN → Generation Scheduler Config.
/// 2. Attach this component to the same GameObject as <see cref="LevelInstantiator"/>
///    (or any persistent object).
/// 3. Assign the config, generator, and level instantiator in the Inspector (or
///    leave blank for auto-discovery).
/// 4. When this component is present and enabled, <see cref="LevelInstantiator"/>
///    automatically delegates all generation scheduling to it.
/// </summary>
public class GenerationScheduler : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("References (auto-discovered if left empty)")]
    public ToadGanGenerator  generator;
    public LevelInstantiator levelInstantiator;

    [Header("Configuration")]
    [Tooltip("ScriptableObject holding all tunable scheduler parameters.")]
    public GenerationSchedulerConfig config;

    // ── Chunk data ────────────────────────────────────────────────────────

    /// <summary>Immutable snapshot of one generated chunk's tile data.</summary>
    public readonly struct ChunkData
    {
        public readonly int[][]                       TileIds;
        public readonly Dictionary<string, string>    TileMap;
        public readonly int                           Height;
        public readonly int                           Width;

        public ChunkData(int[][] tileIds, Dictionary<string, string> tileMap, int height, int width)
        {
            TileIds = tileIds;
            TileMap = tileMap;
            Height  = height;
            Width   = width;
        }
    }

    // ── Runtime state ─────────────────────────────────────────────────────

    private readonly Queue<ChunkData> _chunkBuffer = new Queue<ChunkData>();

    /// <summary>True while a synchronous <c>Generate()</c> call is in flight.</summary>
    private bool _generationInFlight;

    /// <summary>Number of generate calls issued but not yet returned (always 0 or 1 with sync inference).</summary>
    private int _pendingGenerations;

    // ── Timing (exponential moving averages) ──────────────────────────────

    private float _avgInferenceMs;
    private float _avgBuildMs;
    private float _avgChunkWidthTiles;
    private int   _inferenceCount;
    private int   _buildCount;
    private int   _totalChunksGenerated;
    private int   _fallbackChunksSpawned;

    /// <summary>Tracks whether the most recent frontier chunk is a fallback.</summary>
    private bool _fallbackActive;

    // ── Public read-only state for PerformanceMonitor ─────────────────────

    /// <summary>Chunks sitting in the buffer waiting to be built.</summary>
    public int   BufferedChunks          => _chunkBuffer.Count;
    /// <summary>EMA of ML inference duration in milliseconds.</summary>
    public float AvgInferenceMs          => _avgInferenceMs;
    /// <summary>EMA of chunk build duration in milliseconds.</summary>
    public float AvgBuildMs              => _avgBuildMs;
    /// <summary>Current safe-prefetch target computed from timing + speed.</summary>
    public float ComputedSafePrefetch    => ComputeSafePrefetchChunks();
    /// <summary>True while waiting for a Generate() call to return.</summary>
    public bool  IsGenerating            => _generationInFlight;
    /// <summary>True if the last frontier chunk was a fallback.</summary>
    public bool  FallbackActive          => _fallbackActive;
    /// <summary>Total ML chunks generated this session.</summary>
    public int   TotalChunksGenerated    => _totalChunksGenerated;
    /// <summary>Total fallback chunks spawned this session.</summary>
    public int   FallbackChunksSpawned   => _fallbackChunksSpawned;

    // ── Unity lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        if (generator == null)
            generator = FindObjectOfType<ToadGanGenerator>();
        if (levelInstantiator == null)
            levelInstantiator = FindObjectOfType<LevelInstantiator>();

        if (config == null)
        {
            Debug.LogError("[GenerationScheduler] No GenerationSchedulerConfig assigned. Disabling.");
            enabled = false;
            return;
        }
        if (generator == null)
        {
            Debug.LogError("[GenerationScheduler] No ToadGanGenerator found. Disabling.");
            enabled = false;
            return;
        }
        if (levelInstantiator == null)
        {
            Debug.LogError("[GenerationScheduler] No LevelInstantiator found. Disabling.");
            enabled = false;
            return;
        }

        _avgInferenceMs      = config.initialAssumedInferenceMs;
        _avgBuildMs          = config.initialAssumedBuildMs;
        _avgChunkWidthTiles  = 28f;

        generator.OnLevelGenerated    += HandleChunkGenerated;
        generator.OnGenerationCompleted += HandleInferenceTiming;
        generator.OnError             += HandleGeneratorError;

        levelInstantiator.OnChunkBuilt += HandleBuildTiming;

        PreGenerateStartupBuffer();
    }

    private void Update()
    {
        if (levelInstantiator == null || generator == null) return;

        float playerX         = levelInstantiator.GetPlayerX();
        float frontier        = levelInstantiator.NextChunkX;
        float tileSize        = levelInstantiator.tileSize;
        float builtAheadUnits = frontier - playerX;
        float chunkWidthUnits = _avgChunkWidthTiles * tileSize;

        float chunksAheadBuilt = chunkWidthUnits > 0f
            ? builtAheadUnits / chunkWidthUnits
            : 0f;

        float totalAhead = chunksAheadBuilt + _chunkBuffer.Count;
        float targetAhead = Mathf.Max(config.minChunksAhead, ComputeSafePrefetchChunks());

        // ── Feed buffered chunks to the instantiator ──────────────────────
        if (!levelInstantiator.IsBuilding && _chunkBuffer.Count > 0)
        {
            ChunkData next = _chunkBuffer.Dequeue();
            levelInstantiator.ReceiveChunkData(next.TileIds, next.TileMap, next.Height, next.Width);
            _fallbackActive = false;
        }

        // ── Request new ML generation (at most one per frame) ─────────────
        if (totalAhead < targetAhead
            && !_generationInFlight
            && _pendingGenerations < config.maxConcurrentGenerations)
        {
            RequestGeneration();
        }

        // ── Graceful degradation: fallback chunk ──────────────────────────
        // Triggers when:  player is close to frontier, buffer is empty,
        // instantiator is NOT currently building (so we can enqueue), and
        // either a generation is in flight or the buffer simply ran dry.
        if (builtAheadUnits <= config.fallbackTriggerDistance
            && _chunkBuffer.Count == 0
            && !levelInstantiator.IsBuilding
            && !_fallbackActive)
        {
            SpawnFallbackChunk();
        }
    }

    private void OnDestroy()
    {
        if (generator != null)
        {
            generator.OnLevelGenerated      -= HandleChunkGenerated;
            generator.OnGenerationCompleted -= HandleInferenceTiming;
            generator.OnError               -= HandleGeneratorError;
        }

        if (levelInstantiator != null)
            levelInstantiator.OnChunkBuilt -= HandleBuildTiming;
    }

    // ── Core scheduling logic ─────────────────────────────────────────────

    /// <summary>
    /// Computes the minimum number of chunks that must exist ahead of the
    /// player so that a full inference + build cycle can complete before the
    /// player reaches the frontier.
    ///
    /// Formula:
    ///   totalLatency   = avgInference + avgBuild   (milliseconds)
    ///   chunksPerSec   = maxPlayerSpeed / chunkWidthUnits
    ///   chunksConsumed = (totalLatency / 1000) × chunksPerSec
    ///   safePrefetch   = ⌈chunksConsumed⌉ + safetyMargin
    /// </summary>
    private float ComputeSafePrefetchChunks()
    {
        float chunkWidthUnits = _avgChunkWidthTiles * levelInstantiator.tileSize;
        if (chunkWidthUnits <= 0f)
            return config.minChunksAhead;

        float totalLatencyMs  = _avgInferenceMs + _avgBuildMs;
        float totalLatencySec = totalLatencyMs / 1000f;
        float chunksPerSec    = config.assumedMaxPlayerSpeed / chunkWidthUnits;
        float chunksConsumed  = totalLatencySec * chunksPerSec;

        return Mathf.CeilToInt(chunksConsumed) + config.safetyMarginChunks;
    }

    private void RequestGeneration()
    {
        _generationInFlight = true;
        _pendingGenerations++;
        generator.Generate();
    }

    /// <summary>
    /// Front-loads ML generations during Start() so the player begins with a
    /// warm buffer.  Each call is synchronous, so total startup delay ≈
    /// preGenerateCount × avgInferenceTime.
    /// </summary>
    private void PreGenerateStartupBuffer()
    {
        int count = config.preGenerateCount;
        if (count <= 0)
        {
            RequestGeneration();
            return;
        }

        Debug.Log($"[GenerationScheduler] Pre-generating {count} chunk(s) at startup...");
        for (int i = 0; i < count; i++)
            RequestGeneration();

        Debug.Log($"[GenerationScheduler] Startup buffer ready: {_chunkBuffer.Count} chunk(s) buffered.");
    }

    // ── Event handlers ────────────────────────────────────────────────────

    private void HandleChunkGenerated(
        int[][]                       tileIds,
        Dictionary<string, string>    tileMap,
        int                           height,
        int                           width)
    {
        _chunkBuffer.Enqueue(new ChunkData(tileIds, tileMap, height, width));
        _totalChunksGenerated++;

        float alpha = _inferenceCount <= 1 ? 1f : config.emaAlpha;
        _avgChunkWidthTiles = Mathf.Lerp(_avgChunkWidthTiles, width, alpha);

        _generationInFlight = false;
        _pendingGenerations = Mathf.Max(0, _pendingGenerations - 1);
    }

    private void HandleInferenceTiming(float durationMs)
    {
        _inferenceCount++;
        if (_inferenceCount == 1)
            _avgInferenceMs = durationMs;
        else
            _avgInferenceMs = Mathf.Lerp(_avgInferenceMs, durationMs, config.emaAlpha);
    }

    private void HandleBuildTiming(float buildTimeMs, int tileCount)
    {
        _buildCount++;
        if (_buildCount == 1)
            _avgBuildMs = buildTimeMs;
        else
            _avgBuildMs = Mathf.Lerp(_avgBuildMs, buildTimeMs, config.emaAlpha);
    }

    private void HandleGeneratorError(string error)
    {
        _generationInFlight = false;
        _pendingGenerations = Mathf.Max(0, _pendingGenerations - 1);
        Debug.LogError($"[GenerationScheduler] Generator error: {error}");
    }

    // ── Fallback chunk generation ─────────────────────────────────────────

    /// <summary>
    /// Builds a minimal flat-ground chunk procedurally (zero ML cost) so the
    /// player has safe terrain while the real ML generation catches up.
    /// </summary>
    private void SpawnFallbackChunk()
    {
        Dictionary<string, string> tileMap = generator.TileMap;
        if (tileMap == null)
        {
            Debug.LogWarning("[GenerationScheduler] Cannot build fallback — TileMap not available.");
            return;
        }

        int groundId = -1;
        int airId    = -1;
        foreach (var kvp in tileMap)
        {
            if (kvp.Value == "#" && int.TryParse(kvp.Key, out int gid)) groundId = gid;
            if (kvp.Value == "-" && int.TryParse(kvp.Key, out int aid)) airId    = aid;
        }

        if (groundId < 0 || airId < 0)
        {
            Debug.LogWarning("[GenerationScheduler] Cannot build fallback — missing '#' or '-' in vocab.");
            return;
        }

        int width      = config.fallbackChunkWidth;
        int height     = config.fallbackChunkHeight;
        int groundRows = config.fallbackGroundRows;

        var grid = new int[height][];
        for (int r = 0; r < height; r++)
        {
            grid[r] = new int[width];
            int id = r >= height - groundRows ? groundId : airId;
            for (int c = 0; c < width; c++)
                grid[r][c] = id;
        }

        _fallbackActive = true;
        _fallbackChunksSpawned++;

        levelInstantiator.ReceiveChunkData(grid, tileMap, height, width);

        Debug.LogWarning(
            $"[GenerationScheduler] Fallback chunk #{_fallbackChunksSpawned} spawned " +
            $"({width}×{height} flat). ML generator cannot keep up with player speed.");
    }
}
