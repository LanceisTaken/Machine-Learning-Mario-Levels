using UnityEngine;

/// <summary>
/// Cost-aware generation scheduler that sits between the player and the
/// <see cref="ToadGanGenerator"/> / <see cref="LevelInstantiator"/> pair.
///
/// <b>What it does</b>
/// <list type="number">
///   <item>Tracks the player's chunk index and the "frontier" (the furthest
///         chunk that has been fully built).</item>
///   <item>Computes a <em>dynamic prefetch distance</em> based on the
///         measured ML inference time and the player's maximum possible speed
///         so that chunks are always ready before the player reaches them.</item>
///   <item>Enqueues ML generation jobs early enough to keep at least
///         <c>MinChunksAhead</c> built chunks between the player and the
///         frontier.</item>
///   <item>Applies graceful degradation (speed ramp-down) if the model
///         cannot keep up, avoiding a hard stall or void.</item>
/// </list>
///
/// <b>How this mitigates ML inference cost vs hand-authored levels</b>
///
/// Hand-authored / pre-baked levels are loaded from disk or memory in
/// microseconds.  The streaming distance only needs to account for tile
/// instantiation time (~1–3 ms per chunk).  ML-generated levels add a
/// variable <em>inference latency</em> that depends on the model size, GPU
/// backend, and current load.  On integrated GPUs this can exceed 300 ms —
/// enough for the player to cross 5+ tiles at dash speed.
///
/// This scheduler turns the fixed <c>generateAheadDistance</c> into a
/// <em>dynamic</em> prefetch distance that adapts to measured model cost:
///
///   prefetchTiles = max(
///       minChunksAhead × chunkWidth,
///       (avgInferenceMs / 1000) × maxPlayerSpeed × safetyMultiplier
///                                                × maxPendingJobs
///   )
///
/// The formula guarantees that even if the player sprints at maximum speed,
/// the scheduler has already queued enough jobs to cover the worst-case
/// travel distance while those jobs complete.  The <c>safetyMultiplier</c>
/// absorbs variance (spikes in inference time, GC pauses, etc.).
///
/// When the model still falls behind (e.g. a sudden dash burst), the
/// scheduler linearly ramps the player speed down rather than letting
/// them run into empty void, giving the GPU a chance to catch up without
/// a visible freeze.
///
/// Setup
/// -----
///   1. Attach to the same GameObject as <see cref="LevelInstantiator"/>
///      or any persistent object.
///   2. Assign references (or leave null for auto-discovery).
///   3. Create a <see cref="SchedulerConfig"/> asset and drag it in.
///   4. The scheduler takes over generation triggering — the old
///      <c>generateAheadDistance</c> check in LevelInstantiator is
///      automatically bypassed when a ChunkScheduler is present.
/// </summary>
public class ChunkScheduler : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("References (auto-discovered if null)")]
    public ToadGanGenerator  generator;
    public LevelInstantiator levelInstantiator;
    public GameObject        player;

    [Header("Config")]
    [Tooltip("Drag a SchedulerConfig asset here. If null, defaults are used.")]
    public SchedulerConfig config;

    // ── Public read-only state (consumed by PerformanceMonitor) ──────────

    /// <summary>Chunk index the player is currently standing in.</summary>
    public int PlayerChunkIndex { get; private set; }

    /// <summary>Index of the last chunk that has been fully built.</summary>
    public int FrontierChunkIndex { get; private set; } = -1;

    /// <summary>Number of fully-built chunks ahead of the player.</summary>
    public int ChunksAhead => Mathf.Max(0, FrontierChunkIndex - PlayerChunkIndex);

    /// <summary>The current dynamic prefetch distance in tiles.</summary>
    public float PrefetchTiles { get; private set; }

    /// <summary>EMA-smoothed ML inference time in ms.</summary>
    public float SmoothedInferenceMs { get; private set; }

    /// <summary>
    /// Current speed multiplier applied to the player (1 = full speed,
    /// &lt; 1 = being slowed because the frontier is too close).
    /// </summary>
    public float SpeedMultiplier { get; private set; } = 1f;

    /// <summary>Cumulative count of frames where the player was slowed.</summary>
    public int TotalSlowdownFrames { get; private set; }

    /// <summary>
    /// The highest chunk index for which a generation request has been
    /// enqueued (but may not be built yet).
    /// </summary>
    public int HighestRequestedChunk { get; private set; } = -1;

    // ── Runtime state ─────────────────────────────────────────────────────

    private Rigidbody2D _playerRb;
    private PlayerController _playerController;
    private float _originalMoveSpeed;
    private float _originalDashSpeed;
    private float _chunkWidthUnits;
    private bool  _hasSpeeds;

    // Fallback defaults when no config asset is assigned.
    private const int   DefaultMinChunksAhead           = 3;
    private const int   DefaultMaxPendingJobs           = 2;
    private const float DefaultMaxPlayerSpeed           = 18f;
    private const float DefaultSafetyMultiplier         = 1.5f;
    private const float DefaultInitialInferenceMs       = 150f;
    private const float DefaultEmaAlpha                 = 0.2f;
    private const float DefaultFrontierSlowdownFactor   = 0.35f;
    private const float DefaultSlowdownRampTiles        = 5f;

    // Accessors that fall through to defaults when config is null.
    private int   MinChunksAhead         => config != null ? config.minChunksAhead           : DefaultMinChunksAhead;
    private int   MaxPendingJobs         => config != null ? config.maxPendingJobs            : DefaultMaxPendingJobs;
    private float MaxPlayerSpeed         => config != null ? config.maxPlayerSpeed            : DefaultMaxPlayerSpeed;
    private float SafetyMultiplier       => config != null ? config.safetyMultiplier          : DefaultSafetyMultiplier;
    private float EmaAlpha               => config != null ? config.emaAlpha                  : DefaultEmaAlpha;
    private float FrontierSlowdownFactor => config != null ? config.frontierSlowdownFactor    : DefaultFrontierSlowdownFactor;
    private float SlowdownRampTiles      => config != null ? config.slowdownRampTiles         : DefaultSlowdownRampTiles;

    // ── Unity lifecycle ───────────────────────────────────────────────────

    /// <summary>
    /// Awake runs before any Start().  We set the externalScheduler flag here
    /// so LevelInstantiator.Start() sees it and skips its own first-chunk
    /// request regardless of script execution order.
    /// </summary>
    private void Awake()
    {
        if (generator == null)
            generator = FindObjectOfType<ToadGanGenerator>();
        if (levelInstantiator == null)
            levelInstantiator = FindObjectOfType<LevelInstantiator>();

        if (generator == null || levelInstantiator == null)
        {
            Debug.LogError("[ChunkScheduler] Missing generator or levelInstantiator. Disabling.");
            enabled = false;
            return;
        }

        levelInstantiator.externalScheduler = true;
    }

    private void Start()
    {
        if (!enabled) return;

        if (player == null)
            player = levelInstantiator.player;
        if (player != null)
        {
            _playerRb = player.GetComponent<Rigidbody2D>();
            _playerController = player.GetComponent<PlayerController>();
            if (_playerController != null)
            {
                _originalMoveSpeed = _playerController.moveSpeed;
                _originalDashSpeed = _playerController.dashSpeed;
                _hasSpeeds = true;
            }
        }

        float initialMs = config != null ? config.initialInferenceEstimateMs : DefaultInitialInferenceMs;
        SmoothedInferenceMs = initialMs;

        generator.OnGenerationCompleted += OnGenerationCompleted;
        levelInstantiator.OnChunkBuilt  += OnChunkBuilt;

        _chunkWidthUnits = levelInstantiator.LastChunkWidthUnits;
        if (_chunkWidthUnits <= 0f)
            _chunkWidthUnits = 188f * levelInstantiator.tileSize;

        Debug.Log($"[ChunkScheduler] Started. Initial inference estimate: {SmoothedInferenceMs:F0} ms, " +
                  $"chunk width: {_chunkWidthUnits:F0} units.");
    }

    private void Update()
    {
        if (player == null || generator == null || levelInstantiator == null) return;

        UpdatePlayerChunkIndex();
        UpdatePrefetchDistance();
        ScheduleJobs();
        ApplySlowdown();
    }

    private void OnDestroy()
    {
        RestorePlayerSpeed();

        if (generator != null)
            generator.OnGenerationCompleted -= OnGenerationCompleted;
        if (levelInstantiator != null)
            levelInstantiator.OnChunkBuilt -= OnChunkBuilt;
    }

    // ── Event handlers ────────────────────────────────────────────────────

    private void OnGenerationCompleted(float durationMs)
    {
        SmoothedInferenceMs = SmoothedInferenceMs <= 0f
            ? durationMs
            : Mathf.Lerp(SmoothedInferenceMs, durationMs, EmaAlpha);
    }

    private void OnChunkBuilt(float buildTimeMs, int tileCount)
    {
        FrontierChunkIndex++;
        _chunkWidthUnits = levelInstantiator.LastChunkWidthUnits;
    }

    // ── Core scheduling logic ─────────────────────────────────────────────

    private void UpdatePlayerChunkIndex()
    {
        float playerX = _playerRb != null ? _playerRb.position.x : player.transform.position.x;
        if (_chunkWidthUnits > 0f)
            PlayerChunkIndex = Mathf.FloorToInt(playerX / _chunkWidthUnits);
    }

    /// <summary>
    /// Computes how far ahead (in tiles) we need terrain to be ready.
    ///
    /// Formula:
    ///   speed-based = (inferenceMs / 1000) × maxPlayerSpeed × safety × slots
    ///   floor       = minChunksAhead × chunkWidthTiles
    ///   prefetch    = max(floor, speed-based)
    /// </summary>
    private void UpdatePrefetchDistance()
    {
        float chunkWidthTiles = _chunkWidthUnits / Mathf.Max(levelInstantiator.tileSize, 0.01f);
        float floorTiles = MinChunksAhead * chunkWidthTiles;

        float inferenceSeconds = SmoothedInferenceMs / 1000f;
        int   slots            = MaxPendingJobs;
        float speedBased       = inferenceSeconds * MaxPlayerSpeed * SafetyMultiplier * slots;

        PrefetchTiles = Mathf.Max(floorTiles, speedBased);
    }

    private void ScheduleJobs()
    {
        float playerX   = _playerRb != null ? _playerRb.position.x : player.transform.position.x;
        float frontierX = levelInstantiator.NextChunkX;
        float ts        = Mathf.Max(levelInstantiator.tileSize, 0.01f);
        float chunkWidthTiles = _chunkWidthUnits / ts;

        // Built tiles remaining between the player and the current frontier.
        float builtTilesAhead = (frontierX - playerX) / ts;

        // Count ML queue + active inference, AND a chunk currently being
        // instantiated (NextChunkX not advanced yet).  Without the latter,
        // maxPendingJobs=1 still allows "ML finished → immediately request
        // next chunk" while the previous chunk is still building over many
        // frames, which looks like multiple chunks generating ahead at once.
        int mlInflight = generator.PendingJobCount + (generator.IsProcessing ? 1 : 0);
        int buildInflight = levelInstantiator.IsBuildingChunk ? 1 : 0;
        int inflight      = mlInflight + buildInflight;
        float projectedTilesAhead = builtTilesAhead + inflight * chunkWidthTiles;

        while (projectedTilesAhead < PrefetchTiles && inflight < MaxPendingJobs)
        {
            int nextChunk = HighestRequestedChunk + 1;
            generator.RequestGeneration(chunkIndex: nextChunk);
            HighestRequestedChunk = nextChunk;
            inflight++;
            projectedTilesAhead += chunkWidthTiles;
        }
    }

    // ── Graceful degradation ──────────────────────────────────────────────

    private void ApplySlowdown()
    {
        if (!_hasSpeeds) { SpeedMultiplier = 1f; return; }

        float playerX   = _playerRb != null ? _playerRb.position.x : player.transform.position.x;
        float frontierX = levelInstantiator.NextChunkX;
        float tilesToFrontier = (frontierX - playerX) / Mathf.Max(levelInstantiator.tileSize, 0.01f);

        float ramp = SlowdownRampTiles;
        if (ramp <= 0f || tilesToFrontier >= ramp)
        {
            SpeedMultiplier = 1f;
        }
        else
        {
            float t = Mathf.Clamp01(tilesToFrontier / ramp);
            SpeedMultiplier = Mathf.Lerp(FrontierSlowdownFactor, 1f, t);
        }

        _playerController.moveSpeed = _originalMoveSpeed * SpeedMultiplier;
        _playerController.dashSpeed = _originalDashSpeed * SpeedMultiplier;

        if (SpeedMultiplier < 0.99f)
            TotalSlowdownFrames++;
    }

    private void RestorePlayerSpeed()
    {
        if (_hasSpeeds && _playerController != null)
        {
            _playerController.moveSpeed = _originalMoveSpeed;
            _playerController.dashSpeed = _originalDashSpeed;
        }
    }
}
