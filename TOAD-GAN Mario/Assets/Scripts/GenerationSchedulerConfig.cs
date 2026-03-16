using UnityEngine;

/// <summary>
/// Tunable parameters for <see cref="GenerationScheduler"/>.
///
/// Create an instance via Assets → Create → TOAD-GAN → Generation Scheduler Config,
/// then assign it to the <see cref="GenerationScheduler.config"/> field in the Inspector.
/// All values are hot-reloadable at runtime through the Inspector.
/// </summary>
[CreateAssetMenu(fileName = "GenerationSchedulerConfig", menuName = "TOAD-GAN/Generation Scheduler Config")]
public class GenerationSchedulerConfig : ScriptableObject
{
    // ── Prefetch ──────────────────────────────────────────────────────────

    [Header("Prefetch Buffer")]

    [Tooltip("Minimum number of complete chunks (generated + buffered + built) " +
             "to maintain ahead of the player at all times. The scheduler will " +
             "request new ML generations until this target is met.\n\n" +
             "The actual target may be higher if the computed safe-prefetch " +
             "distance (based on inference time × player speed) exceeds this value.")]
    [Range(1, 10)]
    public int minChunksAhead = 3;

    [Tooltip("Maximum generation jobs the scheduler will issue before waiting " +
             "for results. With synchronous Sentis inference this caps at 1 " +
             "(each Generate() blocks until done). Increase when async " +
             "inference (F-06) is implemented.")]
    [Range(1, 4)]
    public int maxConcurrentGenerations = 1;

    [Tooltip("Extra chunks added on top of the speed-based safe-prefetch " +
             "calculation. Acts as a safety margin for timing variance.")]
    [Range(0, 5)]
    public int safetyMarginChunks = 1;

    // ── Player speed assumption ──────────────────────────────────────────

    [Header("Speed Assumption")]

    [Tooltip("Worst-case player speed (units/sec) used to compute how many " +
             "chunks the player can consume during one inference cycle.\n\n" +
             "Set this to the fastest possible movement mode (e.g. dash speed). " +
             "The scheduler will use this to guarantee the buffer never runs " +
             "dry even during a sustained sprint.")]
    public float assumedMaxPlayerSpeed = 18f;

    // ── Graceful degradation ─────────────────────────────────────────────

    [Header("Graceful Degradation")]

    [Tooltip("When the player is within this many world-units of the terrain " +
             "frontier AND the chunk buffer is empty, the scheduler spawns a " +
             "flat emergency fallback chunk instead of stalling.\n\n" +
             "Lower = more aggressive (waits longer before fallback). " +
             "Higher = more conservative (fallback appears earlier).")]
    public float fallbackTriggerDistance = 5f;

    [Tooltip("Tile width of the emergency fallback chunk. Kept short so ML " +
             "chunks can replace the gap quickly.")]
    [Range(8, 32)]
    public int fallbackChunkWidth = 16;

    [Tooltip("Tile height of the fallback chunk (should match the model's " +
             "typical output height — usually 14 for TOAD-GAN).")]
    [Range(8, 20)]
    public int fallbackChunkHeight = 14;

    [Tooltip("Number of solid ground rows at the bottom of the fallback chunk.")]
    [Range(1, 4)]
    public int fallbackGroundRows = 2;

    // ── Timing estimation ────────────────────────────────────────────────

    [Header("Timing Estimation")]

    [Tooltip("Smoothing factor for the exponential moving average of inference " +
             "and build times.  Lower (→ 0.1) = more responsive to recent " +
             "measurements. Higher (→ 0.9) = more stable, slower to react.")]
    [Range(0.05f, 0.95f)]
    public float emaAlpha = 0.3f;

    [Tooltip("Assumed inference time (ms) before any real measurements are " +
             "available.  Used for the initial prefetch calculation. Overwritten " +
             "by the first real measurement.")]
    public float initialAssumedInferenceMs = 300f;

    [Tooltip("Assumed chunk build time (ms) before any real measurements. " +
             "The build coroutine spreads work across frames so this is typically " +
             "small (30–80 ms wall-clock).")]
    public float initialAssumedBuildMs = 50f;

    // ── Pre-generation at startup ────────────────────────────────────────

    [Header("Startup")]

    [Tooltip("Number of chunks to pre-generate synchronously during Start(). " +
             "This front-loads the ML cost into the loading phase so the player " +
             "starts with a comfortable buffer.\n\n" +
             "Each chunk costs ~one inference time (e.g. 200 ms). Set to 0 to " +
             "skip pre-generation and generate reactively.")]
    [Range(0, 6)]
    public int preGenerateCount = 2;
}
