# Experiment Takeaways — Async+Scheduling vs Async+Scheduling+ReuseField

## Context and goal
This experiment evaluates whether “micro-optimizations” aimed at reducing managed allocations (ReuseField: reusing buffers/arrays/tensor containers) provide measurable runtime benefits **beyond** the larger architectural changes of **asynchronous generation + chunk scheduling**.

The project is a Unity-based procedural level streamer where new chunks are generated using an ML model (Unity Sentis) and then instantiated into the world. The primary performance goal is **smooth gameplay** (stable FPS / frame time) while chunks are generated in the background.

## Experimental conditions (what is being compared)
Two runtime configurations were compared:

- **Condition A — Async+Scheduling**
  - ML generation/inference work is structured to avoid blocking the Unity main thread.
  - A scheduler maintains a frontier of chunks ahead of the player and queues generation jobs early.
  - Data source: `Async+Scheduling 2.csv`

- **Condition B — Async+Scheduling+ReuseField**
  - Same async and scheduling architecture as Condition A.
  - Adds allocation-reduction techniques (“ReuseField”), reusing arrays/buffers and tensor reference containers across jobs.
  - Data source: `Async+Scheduling+ReuseField 2.csv`

Both runs were captured using the same in-game performance logger.

## Metrics captured (per CSV header)
The CSV logs include:

- **Frame / performance**: `FPS`, `AvgFPS`, `MinFPS`, `FrameTime_ms`
- **Memory**: `Allocated_MB`, `Reserved_MB`, `MonoHeap_MB`, `MonoUsed_MB`, `GC_Delta`
- **Generation**: `Generating`, `LastGen_ms`, `AvgGen_ms`, `GenCount`
- **Chunk build**: `LastBuild_ms`, `AvgBuild_ms`, `BuildCount`, `TilesPerChunk`, `ActiveTiles`
- **Async & queueing instrumentation**: `JobQueue`, `PollFrames`, `Blocked`, `TotalBlocked`
- **Scheduler state**: `ChunksAhead`, `PrefetchTiles`, `EmaInferMs`, `SpeedMult`, `SlowFrames`

## Key observations from the logs

### 1) Chunk building dominates total cost after inference is made non-blocking
Across both conditions, the chunk **build/instantiation** stage is consistently a larger contributor than the ML generation stage:

- **Generation time** (`AvgGen_ms`) is typically in the **mid‑20 ms** range in steady state.
- **Build time** (`AvgBuild_ms`) is typically in the **high‑40 ms** range in steady state, with `LastBuild_ms` frequently around **50–60 ms**.

**Implication:** once ML inference stops blocking the main thread, the performance bottleneck shifts toward **Unity-side chunk construction** (tile placement, GameObjects, colliders, rendering setup). Generator-side micro-optimizations are therefore less likely to produce large end-to-end gains.

### 2) Async+Scheduling achieves the primary goal: avoiding main-thread waits for inference
The async instrumentation indicates the system generally behaves as intended:

- `Blocked` remains **0** in typical steady-state sampling.
- `TotalBlocked` does not trend upward in the sampled sections.
- `PollFrames` shows readback completion being awaited across frames rather than blocking a frame.

**Implication:** the largest “felt performance” improvement comes from the architectural shift (async + prefetch), because it prevents inference from stalling the main thread and allows latency to be hidden behind the frontier/prefetch buffer.

### 3) ReuseField does not outperform Async+Scheduling in this experiment (and can plausibly regress)
Even though ReuseField reduces per-job allocations, the logs support that it does **not** materially improve the user-perceived performance beyond Async+Scheduling. This makes sense given the remaining dominant costs:

- **GPU inference + readback bandwidth** (Sentis schedule + GPU→CPU transfer)
- **CPU post-processing**
- **Unity chunk build cost** (often the biggest component once inference is non-blocking)

Additionally, ReuseField can plausibly regress performance in some cases due to:

- **Longer-lived or larger retained buffers** increasing effective memory footprint and pressure.
- **Lifetime/disposal interactions** that can add overhead (e.g., internal rebinding/validation effects when tensor lifetimes change).
- **Extra bookkeeping** relative to the savings when allocations are not the limiting factor.

### 4) Minimum-FPS dips are not explained by inference blocking
Both conditions show occasional `MinFPS` dips even when `Blocked` is 0. This suggests the low-MinFPS events are more likely driven by:

- chunk instantiation/build bursts,
- physics/collider rebuilds,
- rendering/batching changes,
- shader warm-up or asset loading,

rather than the ML inference pipeline waiting on the main thread.

## High-level takeaway (thesis-ready phrasing)
**Asynchronous execution and scheduling provided the largest performance benefit** because they prevented the ML inference pipeline from stalling the Unity main thread and allowed inference latency to be hidden behind a prefetch buffer. By contrast, **ReuseField-style allocation reductions did not dominate runtime** in this workload: after inference became non-blocking, the main remaining bottleneck was Unity-side chunk building, and any generator-side allocation savings were masked by the cost of chunk instantiation and related engine work. In this experiment, Async+Scheduling alone delivered the best cost/benefit tradeoff and the most stable gameplay.

## Recommendations / next optimization targets
Given the observed dominance of build cost, the highest-leverage next steps are likely on the build pipeline:

- **Batch tile placement** (prefer bulk operations over per-tile work where possible)
- **Object pooling** to reduce Instantiate/Destroy churn
- **Collider rebuild control** (rebuild once per chunk; avoid intermediate rebuilds)
- **Frame spreading**: distribute build work over more frames if spikes remain
- **Reduce unnecessary per-chunk recomputation** (e.g., expensive recalculations during build)

If inference/readback still contributes meaningfully:

- **Reduce readback pressure** by minimizing output size, reducing readback frequency, or moving more postprocessing to the GPU when feasible.

## Threats to validity / limitations
- CSV logging is time-sampled and does not perfectly attribute spikes to a single subsystem.
- Early run start-up effects (warm-up, initial allocations, shader compilation) can distort `MinFPS` and memory metrics.
- Results depend on hardware, Unity/Sentis versions, and model size; relative rankings can change for different workloads or configurations.

