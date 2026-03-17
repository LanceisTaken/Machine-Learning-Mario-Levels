using UnityEngine;

/// <summary>
/// Tuning parameters for the cost-aware <see cref="ChunkScheduler"/>.
///
/// Create via  Assets ▸ Create ▸ TOAD-GAN ▸ Scheduler Config  and assign to
/// the ChunkScheduler Inspector slot.  Multiple presets can coexist for A/B
/// experiments (e.g. "Conservative", "Aggressive", "LowEnd GPU").
///
/// <b>Why these settings exist (ML cost awareness)</b>
///
/// Unlike hand-authored levels that are loaded from disk in microseconds,
/// TOAD-GAN must run a multi-scale neural network for every chunk.  The
/// inference latency varies by hardware (20 ms on a desktop GPU to 300+ ms
/// on integrated graphics).  These parameters let the scheduler compensate:
///
///   <see cref="minChunksAhead"/>     – hard floor: never fewer than this
///                                      many ready chunks between the player
///                                      and the frontier.
///   <see cref="safetyMultiplier"/>   – how many "worst-case inference
///                                      cycles" to keep buffered beyond the
///                                      minimum, computed from the measured
///                                      average ML inference time.
///   <see cref="maxPlayerSpeed"/>     – the fastest the player can ever
///                                      move (tiles/s), used to translate
///                                      inference time into tile distance.
///   <see cref="maxPendingJobs"/>     – caps how many ML jobs can be queued
///                                      at once, preventing runaway memory
///                                      use if the model is very slow.
/// </summary>
[CreateAssetMenu(fileName = "SchedulerConfig", menuName = "TOAD-GAN/Scheduler Config")]
public class SchedulerConfig : ScriptableObject
{
    [Header("Prefetch Budget")]
    [Tooltip("Absolute minimum number of fully-built chunks that must exist " +
             "ahead of the player at all times.  The scheduler will not stop " +
             "requesting new ML jobs until this floor is met.")]
    [Min(1)]
    public int minChunksAhead = 3;

    [Tooltip("Maximum number of ML generation jobs that may be pending or " +
             "in-flight simultaneously.  Prevents runaway memory use on slow " +
             "hardware.  Set to 1 for strictly sequential generation.")]
    [Range(1, 8)]
    public int maxPendingJobs = 2;

    [Header("Speed-Based Prefetch")]
    [Tooltip("Maximum speed the player can ever reach, in tiles per second.  " +
             "Used to convert measured ML inference time into a safe tile " +
             "distance.  Should be >= dash speed / tileSize.")]
    [Min(1f)]
    public float maxPlayerSpeed = 18f;

    [Tooltip("Multiplier applied to the speed-based prefetch distance.  " +
             "Values > 1 add safety margin; e.g. 1.5 means 'keep 50% more " +
             "buffer than the minimum needed at max speed'.")]
    [Range(1f, 4f)]
    public float safetyMultiplier = 1.5f;

    [Header("Adaptive Inference Estimate")]
    [Tooltip("Initial assumed inference time (ms) used before any real " +
             "measurements are available.  Should be a pessimistic guess " +
             "for the target hardware.")]
    [Min(10f)]
    public float initialInferenceEstimateMs = 150f;

    [Tooltip("Exponential moving average weight for updating the inference " +
             "time estimate.  Lower = smoother but slower to adapt; higher = " +
             "reacts faster to spikes.")]
    [Range(0.05f, 0.5f)]
    public float emaAlpha = 0.2f;

    [Header("Graceful Degradation")]
    [Tooltip("When the player reaches the frontier with no chunk ready, " +
             "slow the player to this fraction of their normal speed.  " +
             "0 = full stop, 1 = no slowdown (the void is just empty air).")]
    [Range(0f, 1f)]
    public float frontierSlowdownFactor = 0.35f;

    [Tooltip("How close to the frontier (in tiles) before the slowdown " +
             "begins to ramp in.  The slowdown is linearly interpolated " +
             "from 1.0 at this distance to frontierSlowdownFactor at 0.")]
    [Min(0f)]
    public float slowdownRampTiles = 5f;
}
