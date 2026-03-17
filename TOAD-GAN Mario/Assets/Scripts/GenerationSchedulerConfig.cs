using UnityEngine;

/// <summary>
/// Tunable configuration for <see cref="GenerationScheduler"/>.
/// Create an instance via Assets ▶ Create ▶ TOAD-GAN ▶ Generation Scheduler Config,
/// then assign it to the scheduler component.
///
/// All values are exposed in the Inspector so you can change them between
/// experiment runs without touching code.
/// </summary>
[CreateAssetMenu(
    menuName = "TOAD-GAN/Generation Scheduler Config",
    fileName = "GenerationSchedulerConfig")]
public class GenerationSchedulerConfig : ScriptableObject
{
    // ── Buffer ─────────────────────────────────────────────────────────────

    [Header("Buffer")]
    [Tooltip("Minimum number of pre-generated (ready-to-build) chunks to keep in " +
             "the queue at all times.  Increase if the player can outrun generation " +
             "at max dash speed.")]
    [Min(1)] public int minChunksAhead = 3;

    [Tooltip("Extra chunks added on top of the dynamically computed prefetch " +
             "requirement.  Acts as a guard against player speed bursts or " +
             "unexpected inference time spikes.")]
    [Min(0)] public int safetyMarginChunks = 1;

    [Tooltip("Maximum number of generation jobs that can be queued or in-flight " +
             "in ToadGanGenerator at once.  Keep at 1 until you have measured " +
             "that the GPU can handle concurrent inference without contention.")]
    [Min(1)] public int maxConcurrentGenerations = 1;

    [Tooltip("Number of chunks to generate during the loading phase before the " +
             "player is able to move.  A higher value gives a larger safety buffer " +
             "at the cost of a longer initial stall.")]
    [Min(0)] public int preGenerateCount = 2;

    // ── Player speed ───────────────────────────────────────────────────────

    [Header("Player Speed")]
    [Tooltip("Worst-case player speed in world units per second used for the " +
             "prefetch distance formula.  Set this to the maximum dash speed so " +
             "the scheduler remains conservative.")]
    public float assumedMaxPlayerSpeed = 18f;

    // ── Graceful degradation ───────────────────────────────────────────────

    [Header("Graceful Degradation")]
    [Tooltip("Distance (world units) from the level frontier at which an emergency " +
             "flat ground chunk is inserted if the ML buffer is empty.  This prevents " +
             "the player falling into a void while generation catches up.")]
    public float fallbackTriggerDistance = 5f;

    [Tooltip("Width in tiles of a procedurally generated fallback chunk.")]
    [Min(4)] public int fallbackChunkWidth = 16;

    [Tooltip("Height in tiles of a procedurally generated fallback chunk.")]
    [Min(4)] public int fallbackChunkHeight = 14;

    // ── Timing EMA ─────────────────────────────────────────────────────────

    [Header("EMA Smoothing")]
    [Tooltip("Alpha for exponential moving averages of inference and build time. " +
             "Lower (e.g. 0.1) = slow to adapt, very smooth. " +
             "Higher (e.g. 0.5) = fast to adapt, noisier.")]
    [Range(0.05f, 0.9f)] public float emaAlpha = 0.3f;

    [Tooltip("Assumed ML inference time (ms) before any measurements are taken.")]
    public float initialAssumedInferenceMs = 300f;

    [Tooltip("Assumed chunk build time (ms) before any measurements are taken.")]
    public float initialAssumedBuildMs = 100f;
}
