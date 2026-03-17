using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Cost-aware chunk generation scheduler for TOAD-GAN Mario.
///
/// How it mitigates the ML model's performance cost
/// ─────────────────────────────────────────────────
/// Hand-authored levels can be streamed instantly (disk read).  ML levels
/// cost 100–500 ms of GPU/CPU time per chunk.  Without a scheduler the game
/// triggers generation reactively — the player gets close to the edge, the
/// main thread pays the inference cost, and the level catches up only after
/// a visible frame hitch.
///
/// This scheduler solves that by:
///   1. Predictive prefetch — measures actual average inference time and
///      player speed, then triggers generation far enough ahead that the
///      chunk is always ready before it is needed.
///   2. Chunk buffer queue — pre-generated tile data is kept in a FIFO
///      queue decoupled from the build coroutine.  The builder drains the
///      queue one chunk at a time; the scheduler refills it continuously.
///   3. Startup pre-generation — <see cref="GenerationSchedulerConfig.preGenerateCount"/>
///      chunks are requested during the loading phase so play starts with a
///      full buffer.
///   4. Graceful fallback — if the player out-runs the buffer (e.g. sudden
///      dash) a flat procedural ground chunk is inserted in microseconds so
///      gameplay never stalls waiting for inference.
///
/// Prefetch distance formula
/// ─────────────────────────
///   safeTiles = ceil((emaInference + emaBuild) / 1000 × maxSpeed / tileSize)
///             + safetyMarginChunks × chunkWidthTiles
///
/// Setup
/// ─────
/// 1. Create a config asset via Assets ▶ Create ▶ TOAD-GAN ▶ Generation Scheduler Config.
/// 2. Add this component to any persistent GameObject.
/// 3. Assign the config asset.  Generator and LevelInstantiator are auto-discovered.
/// 4. Hit Play.  LevelInstantiator detects the scheduler and defers to it.
/// </summary>
public class GenerationScheduler : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("Configuration")]
    [Tooltip("Tunable config.  Create via Assets > Create > TOAD-GAN > Generation Scheduler Config.")]
    public GenerationSchedulerConfig config;

    [Header("References  (auto-discovered if left empty)")]
    public ToadGanGenerator  generator;
    public LevelInstantiator levelInstantiator;

    // ── Public observables (read by PerformanceMonitor) ────────────────────

    /// <summary>Number of fully generated chunks waiting to be built.</summary>
    public int BufferedChunkCount  => _readyQueue.Count;

    /// <summary>Number of generation jobs currently queued or running in ToadGanGenerator.</summary>
    public int InFlightCount =>
        (generator != null ? generator.PendingJobCount + (generator.IsProcessing ? 1 : 0) : 0);

    /// <summary>EMA of most recent inference durations in ms.</summary>
    public float EmaInferenceMs    { get; private set; }

    /// <summary>EMA of most recent chunk build durations in ms.</summary>
    public float EmaBuildMs        { get; private set; }

    /// <summary>
    /// Computed safe prefetch distance in world units.
    /// Chunks should be queued at least this far ahead of the player.
    /// </summary>
    public float SafePrefetchDistanceUnits { get; private set; }

    /// <summary>Total ML chunks generated this session (excludes fallbacks).</summary>
    public int TotalMlChunksGenerated { get; private set; }

    /// <summary>Total fallback chunks inserted this session.</summary>
    public int TotalFallbacksInserted { get; private set; }

    /// <summary>True while the scheduler is still filling the startup buffer.</summary>
    public bool IsPreGenerating { get; private set; }

    // ── Internal chunk data record ─────────────────────────────────────────

    private struct ChunkData
    {
        public int[][]                   TileIds;
        public Dictionary<string, string> TileMap;
        public int                       Height;
        public int                       Width;
        public bool                      IsFallback;
    }

    // ── State ──────────────────────────────────────────────────────────────

    private readonly Queue<ChunkData> _readyQueue = new Queue<ChunkData>();

    private int   _lastChunkWidthTiles  = 0;
    private int   _lastChunkHeightTiles = 0;
    private bool  _fallbackPending      = false; // fallback triggered; don't repeat

    // Fallback tile ID cache (populated once from TileMap)
    private int _groundTileId = -1;
    private int _airTileId    = -1;
    private bool _tileIdsResolved = false;

    // Player speed tracking (EMA)
    private float _emaPlayerSpeed;
    private float _lastPlayerX = float.MinValue;

    // Default config values used when config asset is null
    private const int   DefaultMinChunksAhead          = 3;
    private const int   DefaultSafetyMargin             = 1;
    private const int   DefaultMaxConcurrent            = 1;
    private const int   DefaultPreGenerateCount         = 2;
    private const float DefaultAssumedMaxSpeed          = 18f;
    private const float DefaultFallbackTriggerDistance  = 5f;
    private const int   DefaultFallbackWidth            = 16;
    private const int   DefaultFallbackHeight           = 14;
    private const float DefaultEmaAlpha                 = 0.3f;
    private const float DefaultInitialInferenceMs       = 300f;
    private const float DefaultInitialBuildMs           = 100f;

    // ── Config accessors (fallback to defaults when config is null) ─────────

    private int   MinChunksAhead           => config != null ? config.minChunksAhead           : DefaultMinChunksAhead;
    private int   SafetyMarginChunks       => config != null ? config.safetyMarginChunks       : DefaultSafetyMargin;
    private int   MaxConcurrentGenerations => config != null ? config.maxConcurrentGenerations : DefaultMaxConcurrent;
    private int   PreGenerateCount         => config != null ? config.preGenerateCount         : DefaultPreGenerateCount;
    private float AssumedMaxPlayerSpeed    => config != null ? config.assumedMaxPlayerSpeed    : DefaultAssumedMaxSpeed;
    private float FallbackTriggerDistance  => config != null ? config.fallbackTriggerDistance  : DefaultFallbackTriggerDistance;
    private int   FallbackChunkWidth       => config != null ? config.fallbackChunkWidth       : DefaultFallbackWidth;
    private int   FallbackChunkHeight      => config != null ? config.fallbackChunkHeight      : DefaultFallbackHeight;
    private float EmaAlpha                 => config != null ? config.emaAlpha                 : DefaultEmaAlpha;

    // ── Unity lifecycle ────────────────────────────────────────────────────

    private IEnumerator Start()
    {
        // Wait one frame so every other Start() method completes first
        // (in particular LevelInstantiator.Start(), which needs to run before
        // we call ReceiveChunkData on it).
        yield return null;

        // Auto-discover references
        if (generator        == null) generator        = FindObjectOfType<ToadGanGenerator>();
        if (levelInstantiator == null) levelInstantiator = FindObjectOfType<LevelInstantiator>();

        if (generator == null)
        {
            Debug.LogError("[GenerationScheduler] ToadGanGenerator not found — scheduler disabled.");
            enabled = false;
            yield break;
        }

        if (levelInstantiator == null)
        {
            Debug.LogError("[GenerationScheduler] LevelInstantiator not found — scheduler disabled.");
            enabled = false;
            yield break;
        }

        // Initialise EMA with assumed values
        float initInference = config != null ? config.initialAssumedInferenceMs : DefaultInitialInferenceMs;
        float initBuild     = config != null ? config.initialAssumedBuildMs     : DefaultInitialBuildMs;
        EmaInferenceMs = initInference;
        EmaBuildMs     = initBuild;

        // Subscribe to generator and instantiator events
        generator.OnLevelGenerated     += OnLevelGenerated;
        generator.OnGenerationCompleted += OnGenerationCompleted;
        levelInstantiator.OnChunkBuilt += OnChunkBuilt;

        UpdateSafePrefetchDistance();

        // Pre-generate startup buffer
        if (PreGenerateCount > 0)
        {
            IsPreGenerating = true;
            for (int i = 0; i < PreGenerateCount; i++)
                RequestIfSlotAvailable();
        }

        Debug.Log($"[GenerationScheduler] Initialised.  " +
                  $"minChunksAhead={MinChunksAhead}, preGenerateCount={PreGenerateCount}, " +
                  $"assumedMaxSpeed={AssumedMaxPlayerSpeed} u/s");
    }

    private void Update()
    {
        if (generator == null || levelInstantiator == null) return;

        TrackPlayerSpeed();
        UpdateSafePrefetchDistance();

        // Stop the pre-generating flag once initial buffer is ready
        if (IsPreGenerating && _readyQueue.Count >= PreGenerateCount && InFlightCount == 0)
            IsPreGenerating = false;

        // Feed the build pipeline
        TryFeedNextChunk();

        // Ensure generation keeps the buffer topped up
        EvaluateAndRequestGenerations();

        // Graceful degradation fallback
        CheckFallback();
    }

    private void OnDestroy()
    {
        if (generator != null)
        {
            generator.OnLevelGenerated      -= OnLevelGenerated;
            generator.OnGenerationCompleted -= OnGenerationCompleted;
        }

        if (levelInstantiator != null)
            levelInstantiator.OnChunkBuilt -= OnChunkBuilt;
    }

    // ── Event handlers ─────────────────────────────────────────────────────

    private void OnLevelGenerated(
        int[][] tileIds, Dictionary<string, string> tileMap, int height, int width)
    {
        _readyQueue.Enqueue(new ChunkData
        {
            TileIds    = tileIds,
            TileMap    = tileMap,
            Height     = height,
            Width      = width,
            IsFallback = false
        });

        _lastChunkWidthTiles  = width;
        _lastChunkHeightTiles = height;
        TotalMlChunksGenerated++;

        // Resolve tile IDs for fallback from the first real TileMap we receive
        if (!_tileIdsResolved)
            ResolveFallbackTileIds(tileMap);

        // Feed immediately if builder is idle
        TryFeedNextChunk();
    }

    private void OnGenerationCompleted(float durationMs)
    {
        EmaInferenceMs = Mathf.Lerp(EmaInferenceMs, durationMs, EmaAlpha);
        UpdateSafePrefetchDistance();
    }

    private void OnChunkBuilt(float buildTimeMs, int tileCount)
    {
        EmaBuildMs = Mathf.Lerp(EmaBuildMs, buildTimeMs, EmaAlpha);
        UpdateSafePrefetchDistance();

        // Reset fallback flag — a real chunk was just built
        _fallbackPending = false;

        // Immediately pull the next queued chunk into the build coroutine
        TryFeedNextChunk();

        // Check if more generations are needed
        EvaluateAndRequestGenerations();
    }

    // ── Core scheduler logic ───────────────────────────────────────────────

    /// <summary>
    /// Feeds the oldest buffered chunk to <see cref="LevelInstantiator"/>
    /// if it is not currently building.
    /// </summary>
    private void TryFeedNextChunk()
    {
        if (levelInstantiator.IsBuilding) return;
        if (_readyQueue.Count == 0) return;

        ChunkData chunk = _readyQueue.Dequeue();
        levelInstantiator.ReceiveChunkData(
            chunk.TileIds, chunk.TileMap, chunk.Height, chunk.Width);
    }

    /// <summary>
    /// Checks whether more generation jobs should be requested and fires
    /// them if the concurrency limit allows.
    /// </summary>
    private void EvaluateAndRequestGenerations()
    {
        // Total "chunks in pipeline" = ready in queue + in-flight in generator
        int inPipeline = _readyQueue.Count + InFlightCount;

        // We want at least minChunksAhead plus one extra per computed prefetch need
        int targetBuffer = Mathf.Max(MinChunksAhead, ComputePrefetchCountNeeded())
                         + SafetyMarginChunks;

        while (inPipeline < targetBuffer && InFlightCount < MaxConcurrentGenerations)
        {
            RequestIfSlotAvailable();
            inPipeline++;
        }
    }

    /// <summary>
    /// Inserts a flat procedural chunk if the player is about to reach the
    /// level frontier with no chunk ready to build.
    /// </summary>
    private void CheckFallback()
    {
        if (_fallbackPending) return;
        if (levelInstantiator.IsBuilding) return;
        if (_readyQueue.Count > 0) return;

        float playerX   = levelInstantiator.GetPlayerX();
        float frontierX = levelInstantiator.NextChunkX;

        if (playerX + FallbackTriggerDistance >= frontierX)
        {
            Debug.LogWarning("[GenerationScheduler] Player approaching frontier with empty " +
                             "buffer — inserting fallback chunk.");
            InsertFallbackChunk();
            _fallbackPending = true;
        }
    }

    // ── Generation request helper ──────────────────────────────────────────

    private void RequestIfSlotAvailable()
    {
        if (InFlightCount >= MaxConcurrentGenerations) return;
        generator.RequestGeneration(TotalMlChunksGenerated);
    }

    // ── Prefetch calculation ───────────────────────────────────────────────

    /// <summary>
    /// How many chunks the player can consume during one full generation + build cycle.
    /// Rounds up so we always err on the side of having more buffered.
    /// </summary>
    private int ComputePrefetchCountNeeded()
    {
        float tileSize   = levelInstantiator.TileSize;
        float chunkUnits = _lastChunkWidthTiles > 0
            ? _lastChunkWidthTiles * tileSize
            : FallbackChunkWidth  * tileSize;

        if (chunkUnits <= 0f) return MinChunksAhead;

        // Time to generate + build one chunk
        float cycleSec  = (EmaInferenceMs + EmaBuildMs) / 1000f;
        float speedUsed = Mathf.Max(AssumedMaxPlayerSpeed, _emaPlayerSpeed);

        // World units the player travels in one cycle
        float distanceTravelled = cycleSec * speedUsed;

        return Mathf.CeilToInt(distanceTravelled / chunkUnits);
    }

    private void UpdateSafePrefetchDistance()
    {
        float tileSize  = levelInstantiator != null ? levelInstantiator.TileSize : 1f;
        float speedUsed = Mathf.Max(AssumedMaxPlayerSpeed, _emaPlayerSpeed);
        float cycleSec  = (EmaInferenceMs + EmaBuildMs) / 1000f;

        SafePrefetchDistanceUnits = cycleSec * speedUsed
            + SafetyMarginChunks * (_lastChunkWidthTiles > 0
                ? _lastChunkWidthTiles * tileSize
                : FallbackChunkWidth  * tileSize);
    }

    // ── Player speed tracking ──────────────────────────────────────────────

    private void TrackPlayerSpeed()
    {
        if (levelInstantiator == null) return;

        float playerX = levelInstantiator.GetPlayerX();
        if (_lastPlayerX < float.MinValue / 2f)
        {
            _lastPlayerX = playerX;
            return;
        }

        float dt = Time.deltaTime;
        if (dt <= 0f) return;

        float speed = Mathf.Abs(playerX - _lastPlayerX) / dt;
        _emaPlayerSpeed = Mathf.Lerp(_emaPlayerSpeed, speed, EmaAlpha);
        _lastPlayerX    = playerX;
    }

    // ── Fallback chunk ─────────────────────────────────────────────────────

    /// <summary>
    /// Builds a simple flat-ground chunk and feeds it directly to the instantiator.
    /// Uses the standard tile IDs from the most recent real TileMap so it integrates
    /// seamlessly with the existing collider and prefab pipeline.
    /// </summary>
    private void InsertFallbackChunk()
    {
        int w = _lastChunkWidthTiles  > 0 ? _lastChunkWidthTiles  : FallbackChunkWidth;
        int h = _lastChunkHeightTiles > 0 ? _lastChunkHeightTiles : FallbackChunkHeight;

        int groundId = _groundTileId >= 0 ? _groundTileId : 0;
        int airId    = _airTileId    >= 0 ? _airTileId    : 1;

        var grid = new int[h][];
        for (int row = 0; row < h; row++)
        {
            grid[row] = new int[w];
            for (int col = 0; col < w; col++)
                grid[row][col] = row == h - 1 ? groundId : airId;
        }

        // Use the last real TileMap so the existing prefab pipeline works
        var tileMap = generator != null ? generator.TileMap : null;

        TotalFallbacksInserted++;

        levelInstantiator.ReceiveChunkData(grid, tileMap, h, w);
        Debug.Log($"[GenerationScheduler] Fallback chunk inserted " +
                  $"({w}×{h}t, total fallbacks: {TotalFallbacksInserted}).");
    }

    private void ResolveFallbackTileIds(Dictionary<string, string> tileMap)
    {
        if (tileMap == null) return;

        foreach (var kvp in tileMap)
        {
            if (!int.TryParse(kvp.Key, out int id)) continue;
            if (kvp.Value == "#" && _groundTileId < 0) _groundTileId = id;
            if (kvp.Value == "-" && _airTileId    < 0) _airTileId    = id;
        }

        _tileIdsResolved = true;
    }
}
