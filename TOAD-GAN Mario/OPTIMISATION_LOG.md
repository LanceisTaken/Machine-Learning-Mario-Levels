# TOAD-GAN Mario – Optimisation Log

Tracks every performance improvement made to the Unity game project.
Each entry records the problem, root cause, fix applied, and the expected/measured impact.

---

## Session 1 — 2026-03-09

**Baseline data**: `perf_log_20260309_173731.csv`

| Metric (baseline) | Value |
|---|---|
| Normal FPS | 200–265 |
| Random stutter FPS | 16–41 FPS (every 4–8 s) |
| Stutter frame time | 24–61 ms |
| Chunk build time | 206–305 ms |
| Mono heap growth rate | ~3 MB/s |

---

### OPT-01 · `LevelInstantiator` — Eliminate per-frame Transform enumerator allocation

**File**: `Assets/Scripts/LevelInstantiator.cs`

**Problem**  
`CleanupOldTiles()` ran `foreach (Transform child in transform)` every frame.
Unity's Transform enumerator allocates a new managed `IEnumerator` object each call.
At 250 FPS this produces ~250 allocations/second, steadily pressuring the GC.
Combined with other allocators, `MonoUsed` grew ~3 MB/s and triggered multi-frame
GC pauses (observed: 24–61 ms drops every 4–8 seconds in the baseline log).

**Fix**  
Added `private readonly List<Transform> _activeTiles` to track all live tile
Transforms. Tiles are added to the list on instantiation and removed when
destroyed. `CleanupOldTiles` now iterates this list directly (backwards for
safe `RemoveAt`) instead of walking the Transform hierarchy.

**Impact**  
Eliminates ~250 IEnumerator allocations per second. Reduces the rate of GC
collection triggers, which should remove or substantially shorten the random
FPS dips between chunk generations.

---

### OPT-02 · `LevelInstantiator` — Switch CompositeCollider2D to Manual geometry generation

**File**: `Assets/Scripts/LevelInstantiator.cs`

**Problem**  
`CompositeCollider2D` was left at its default `Synchronous` generation mode.
Every time a tile's `BoxCollider2D.compositeOperation` was set to `Merge` during
`BuildChunk`, Unity immediately rebuilt the entire composite mesh. With ~350 static
terrain tiles per chunk this caused ~350 incremental full-mesh rebuilds in a single
frame — accounting for most of the 206–305 ms chunk build stall.

**Fix**  
Set `_composite.generationType = CompositeCollider2D.GenerationType.Manual` in
`Start()`. `BuildChunkCoroutine` places all tiles first (with their colliders set
to `Merge`) and calls `_composite.GenerateGeometry()` exactly once after the last
tile is placed, replacing ~350 rebuilds with one.

**Impact**  
Expected to reduce the per-chunk rebuild cost from ~170+ ms (350 × ~0.5 ms each)
to a single rebuild pass (estimated 10–30 ms depending on tile count).

---

### OPT-03 · `LevelInstantiator` — Spread chunk instantiation across frames

**File**: `Assets/Scripts/LevelInstantiator.cs`

**Problem**  
`BuildChunk` ran as a plain method, instantiating all tiles synchronously in one
frame. Even after OPT-02, `Instantiate` itself has overhead (~0.1–0.2 ms/call)
and calling it 350 times still contributes a noticeable frame spike.

**Fix**  
Converted `BuildChunk` to `BuildChunkCoroutine` (IEnumerator). It yields
`null` (waits one frame) after every `tilesPerFrameBudget` tiles (default: 25).
This spreads 350 tile instantiations over ~14 frames (~56 ms real time at 250 FPS)
instead of one large blocking frame.

`GenerateGeometry()` is called only once at the very end of the coroutine,
after all tiles are placed, so the collider is always complete before the player
can reach the new chunk (guaranteed by the larger `generateAheadDistance`).

`generateAheadDistance` default increased from **10 → 20 tiles** to give the
coroutine plenty of build time before the player arrives at the chunk boundary.

**Impact**  
Converts a 200+ ms single-frame stall into a smooth ~14-frame build at
≤10 ms/frame additional overhead (~90–100 FPS during build instead of a
hard freeze).

---

### OPT-04 · `PowerUpItem` — Replace `OverlapCircleAll` with NonAlloc variant

**File**: `Assets/Scripts/PowerUpItem.cs`

**Problem**  
`PowerUpItem.Update()` called `Physics2D.OverlapCircleAll(...)` every frame.
This method always allocates and returns a new `Collider2D[]` array, even when
no overlap is found. With power-ups active this produced a heap allocation every
frame per live power-up, contributing to the GC pressure measured in the baseline.

**Fix**  
Added a `private static readonly Collider2D[] _overlapBuffer = new Collider2D[4]`
field (static so it is shared across all `PowerUpItem` instances and allocated
only once at class load). Replaced `OverlapCircleAll` with
`Physics2D.OverlapCircleNonAlloc(..., _overlapBuffer)` which writes results into
the existing buffer instead of allocating a new array.

**Impact**  
Zero managed heap allocations per frame for power-up pickup detection.
Buffer size of 4 safely covers any realistic scenario (player + collectibles in range).

---

## Session 2 — 2026-03-09

**Post-Session-1 data**: `perf_log_20260309_175202.csv`

| Metric (after Session 1) | Value |
|---|---|
| Normal FPS | 400–500 (huge improvement) |
| Chunk build stutter | Eliminated (coroutine) |
| Random stutter FPS | 41–46 FPS (still every ~7 s) |
| Mono heap growth rate | ~4–5 MB/s (still high) |
| GC_Delta | 3 every ~7 seconds |
| Progressive FPS degradation | 450 → 200 → 100 over 2 minutes |

Session 1 fixed the chunk generation stall, but periodic GC pauses remained.
Profiling the remaining allocation sources revealed three major culprits that
were active every frame even while standing still.

---

### OPT-05 · `PerformanceMonitor` — Cache all GUIStyles (eliminated ~28 allocations/frame)

**File**: `Assets/Scripts/PerformanceMonitor.cs`

**Problem**  
`DrawSectionLabel()` and `DrawRow()` both created **brand new `GUIStyle`
objects** on every call:

```csharp
// OLD — new GUIStyle on EVERY call
GUIStyle s = new GUIStyle(_labelStyle) { ... };
```

`OnGUI` is called at least twice per frame (Layout + Repaint), invoking
`DrawRow` 11 times and `DrawSectionLabel` 3 times per pass.  
That's **(11 + 3) × 2 = 28 GUIStyle heap allocations per frame**.

At 450 FPS this produced ~12,600 managed allocations/second — easily the
single largest source of the ~4–5 MB/s GC pressure measured in the post-
Session-1 log.

**Fix**  
Moved both styles (`_sectionStyle`, `_rowLabelStyle`) into `EnsureStyles()`,
where they are created exactly once alongside the other cached styles.
`DrawSectionLabel` and `DrawRow` now reference the pre-built fields.

**Impact**  
Eliminates ~12,600 GUIStyle allocations/second. This was the dominant source
of GC pressure while idle — expected to dramatically reduce the frequency
and severity of GC pause frame drops.

---

### OPT-06 · `PerformanceMonitor` — Replace `Queue<float>` FPS history with zero-alloc circular buffer

**File**: `Assets/Scripts/PerformanceMonitor.cs`

**Problem**  
`TrackFps()` used `foreach (float f in _fpsHistory)` on a `Queue<float>`.
The `Queue<T>.GetEnumerator()` call allocates a new managed enumerator object
each frame — one more per-frame allocation contributing to GC pressure.

**Fix**  
Replaced `Queue<float> _fpsHistory` with a pre-allocated `float[] _fpsRing`
circular buffer. `_fpsRingHead` advances modulo the capacity. Stats are
computed with a `for (int i = 0; ...)` loop over the raw array — zero
allocation per frame.

**Impact**  
Eliminates 1 managed allocation per frame from FPS tracking.

---

### OPT-07 · `PlayerController` — Replace `OverlapCircleAll` with NonAlloc in dash kill detection

**File**: `Assets/Scripts/PlayerController.cs`

**Problem**  
`KillEnemiesInRange()` (called every frame during a dash) used
`Physics2D.OverlapCircleAll(...)`, which allocates a new `Collider2D[]` each
call — the same issue previously fixed in `PowerUpItem` (OPT-04).

**Fix**  
Added `private static readonly Collider2D[] _dashOverlapBuffer = new Collider2D[8]`.
Replaced `OverlapCircleAll` with `OverlapCircleNonAlloc`.

**Impact**  
Zero allocation per frame during dash. Buffer size of 8 covers dense enemy
scenarios.

---

### OPT-08 · `PerformanceMonitor` — Gate FPS-drop warning behind `UNITY_EDITOR`

**File**: `Assets/Scripts/PerformanceMonitor.cs`

**Problem**  
`Debug.LogWarning($"... {_currentFps:F1} ...")` ran in builds whenever FPS
dropped below threshold. Each call performs string interpolation (heap
allocation) and the `Debug.Log` pipeline itself has overhead in release
builds.

**Fix**  
Wrapped the FPS-drop warning in `#if UNITY_EDITOR` so it is stripped from
non-editor builds entirely.

**Impact**  
Removes a periodic string allocation + log call during exactly the moments
when the game is already under GC stress (FPS drop events).

---

## Session 3 — 2026-03-09

**Symptom**: FPS drops below 60 **instantly** the moment the player presses any movement key (left, right, or jump). Returns to high FPS when standing still. Affects all directions including jump (i.e. it is not related to horizontal chunk generation).

---

### OPT-09 · `LevelInstantiator` — Eliminate `Transform.position` reads in `CleanupOldTiles`

**File**: `Assets/Scripts/LevelInstantiator.cs`

**Problem**  
`CleanupOldTiles()` previously read `t.position.x` for every tile in the
`_activeTiles` list (up to ~700 entries) on every `Update()` frame.

`Transform.position` is a native C++ property — each access crosses the
managed/native boundary. More critically: when the player is moving, Unity's
**physics job system** is active (the player's `Rigidbody2D` is simulating).
Reading transform positions from the main thread while physics jobs are
running forces a **job-sync stall** — the main thread blocks until the current
physics batch finishes before it can return the value.

At 400+ FPS with ~700 tiles:
- **Standing still**: physics jobs idle → reads return immediately → no stall → 400+ FPS
- **First frame of movement**: physics jobs start → 700 reads × job-sync stall → instant 50-100 ms stall → drops well below 60 FPS

This explained the exact symptom: high FPS when idle, instant drop the moment any input starts physics activity.

**Fix**  
Added a parallel `List<float> _activeTileXs` whose entries are written at
tile creation time from the already-computed `float x` local variable.
Static tiles never move, so this cached value is always accurate.

`CleanupOldTiles` now compares `_activeTileXs[i] < cutoff` — a plain managed
float list read, zero native overhead, zero job-sync dependency.

**Impact**  
Eliminates the physics job-sync stall from the hot path. The instant FPS drop
on movement start should disappear.

---

### OPT-10 · `LevelInstantiator` — Gate `CleanupOldTiles` behind a movement threshold

**File**: `Assets/Scripts/LevelInstantiator.cs`

**Problem**  
Even with cached X positions, `CleanupOldTiles` still iterated all ~700
entries every frame (now as float comparisons, which are fast). On frames
where the player has barely moved, no tiles can have crossed the cutoff, so
the entire loop produces zero results — wasted work.

**Fix**  
Added `_lastCleanupPlayerX` and a `CleanupMoveThreshold` constant (0.25 units).
The loop returns immediately with a single `float` subtraction + comparison if
the player hasn't moved at least 0.25 units since the last pass.

At typical walk speed (6 u/s) this means the loop actually runs ~24 times/second
instead of 400+ times/second — a ~16x reduction in loop executions.
Reading `player.transform.position.x` (one read, the player's own transform,
not a static tile) remains fine since the player's transform is not
job-system-managed the same way.

**Impact**  
Reduces CleanupOldTiles CPU cost by ~16× at normal walk speed. At high FPS
the loop was running hundreds of times per second accomplishing nothing.

---

### OPT-11 · `PerformanceMonitor` — Pre-compute color hex strings (eliminate `ColorUtility.ToHtmlStringRGB` allocations)

**File**: `Assets/Scripts/PerformanceMonitor.cs`

**Problem**  
`OnGUI` called `ColorHex(FpsColor(fps))` multiple times per pass.
`ColorHex` called `ColorUtility.ToHtmlStringRGB(c)`, which allocates a new
`string` on every call. `OnGUI` runs ≥2 passes per frame, with 5–6 color
hex calls per pass:

- 5–6 calls × 2 passes × 400 FPS = **4,000–4,800 string allocations/second**
  just for color hex strings alone.

**Fix**  
Defined four `private static readonly string` fields (`HexGreen`, `HexYellow`,
`HexRed`, `HexBlue`) computed at class load via `ColorUtility.ToHtmlStringRGB`.
Replaced `ColorHex(FpsColor(...))` and `ColorHex(FrameTimeColor(...))` with
`FpsColorHex(fps)` and `FrameTimeColorHex(ms)` — static methods that return
the pre-computed strings with no allocation.

`gcColor` was also changed from `Color.yellow / Color.green` → `HexYellow / HexGreen`,
removing two implicit `Color` struct creations and one `ColorHex` call.

**Impact**  
Eliminates ~4,000–4,800 managed string allocations/second from the HUD overlay.

---

## Session 4 — 2026-03-16

**Goal**: Introduce a cost-aware generation scheduler that is explicitly aware of
the ML model's inference cost and the player's position / speed, replacing the
reactive `if (player near edge) → generate()` pattern.

---

### OPT-12 · `GenerationScheduler` — Predictive ML-cost-aware chunk scheduling

**Files**: `Assets/Scripts/GenerationScheduler.cs` (new),
`Assets/Scripts/GenerationSchedulerConfig.cs` (new),
`Assets/Scripts/LevelInstantiator.cs` (modified),
`Assets/Scripts/ToadGanGenerator.cs` (modified),
`Assets/Scripts/PerformanceMonitor.cs` (modified)

**Problem**

The previous generation trigger was purely reactive — `LevelInstantiator.Update()`
checked whether the player was within `generateAheadDistance` tiles of the level
edge and, if so, called `generator.Generate()`.  This had three cost-related
weaknesses:

1. **No awareness of inference duration.**  The trigger distance was a static tile
   count (`generateAheadDistance = 20`) with no relationship to how long ML
   inference actually takes.  If inference is slow (e.g. CPU backend, 400 ms), the
   buffer is too thin; if fast (GPU, 100 ms), it's wastefully thick.

2. **No awareness of player speed.**  A walking player (6 u/s) and a dashing
   player (18 u/s) consume terrain at 3× different rates, but the trigger distance
   was the same for both.  A sustained dash could outrun the buffer and hit the
   void.

3. **No buffer / queue.**  At most one generation was ever in-flight; the system
   could not pre-build a reserve of chunks to absorb speed bursts.

4. **No graceful degradation.**  If the player reached the frontier before a chunk
   was ready, the only outcome was a main-thread stall or falling into empty space.

**Fix — Generation Scheduler**

A new `GenerationScheduler` MonoBehaviour sits between `ToadGanGenerator` and
`LevelInstantiator`.  When present and enabled, `LevelInstantiator` automatically
detects it and disables its own generation trigger (`_schedulerManaged = true`).

**Core formula — safe prefetch distance:**

```
totalLatency   = EMA(inferenceTime) + EMA(buildTime)     [ms]
chunksPerSec   = assumedMaxPlayerSpeed / (avgChunkWidth × tileSize)
chunksConsumed = (totalLatency / 1000) × chunksPerSec
safePrefetch   = ⌈chunksConsumed⌉ + safetyMargin
target         = max(minChunksAhead, safePrefetch)
```

Example with measured values:

| Parameter | Value |
|---|---|
| EMA inference | 200 ms |
| EMA build | 50 ms |
| Max player speed (dash) | 18 u/s |
| Avg chunk width | 28 tiles × 1 u/tile = 28 u |
| Safety margin | 1 chunk |

```
chunksPerSec   = 18 / 28 ≈ 0.643
chunksConsumed = 0.250 × 0.643 ≈ 0.161
safePrefetch   = ⌈0.161⌉ + 1 = 2 chunks
target         = max(3, 2) = 3 chunks
```

So with `minChunksAhead = 3`, the scheduler maintains at least 3 chunks of
terrain ahead.  If inference time increases (e.g. CPU fallback), the formula
automatically raises the target.

**Four mitigation strategies:**

| Strategy | How it mitigates ML cost |
|---|---|
| **Predictive prefetch** | Uses real measured inference time (EMA) × worst-case player speed to start generation *before* it's urgent, so the synchronous stall occurs while the player still has runway. |
| **Startup pre-generation** | Front-loads `preGenerateCount` chunks into a buffer during `Start()`, shifting the ML cost to the loading phase where delay is expected. |
| **Chunk buffer queue** | Decouples "generate" from "build" — ML chunks queue up and are fed to `LevelInstantiator` one at a time.  Multiple chunks can be pre-generated while the first is still being built. |
| **Graceful fallback** | If the player out-runs the buffer (sprint/dash), a flat procedural "fallback chunk" (zero ML cost, microseconds to create) spawns safe ground.  The player never falls into void; the next ML chunk replaces the gap seamlessly. |

**Config (ScriptableObject `GenerationSchedulerConfig`):**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `minChunksAhead` | int | 3 | Hard minimum chunk buffer |
| `maxConcurrentGenerations` | int | 1 | Generation concurrency cap (future async support) |
| `safetyMarginChunks` | int | 1 | Extra margin on computed prefetch |
| `assumedMaxPlayerSpeed` | float | 18 | Worst-case speed (dash) for prefetch calc |
| `fallbackTriggerDistance` | float | 5 | World-units before frontier to spawn fallback |
| `fallbackChunkWidth` | int | 16 | Width of emergency fallback chunk |
| `fallbackChunkHeight` | int | 14 | Height of emergency fallback chunk |
| `fallbackGroundRows` | int | 2 | Solid rows at bottom of fallback |
| `emaAlpha` | float | 0.3 | Smoothing for inference / build time EMA |
| `initialAssumedInferenceMs` | float | 300 | Assumed inference time before first measurement |
| `initialAssumedBuildMs` | float | 50 | Assumed build time before first measurement |
| `preGenerateCount` | int | 2 | Chunks pre-generated at startup |

All values are live-editable in the Inspector during Play for rapid experimentation.

**LevelInstantiator changes:**
- Renamed `_isGenerating` → `_isBuilding` (it always meant "build in progress").
- Exposed `NextChunkX`, `IsBuilding`, `GetPlayerX()` read-only for the scheduler.
- Added `ReceiveChunkData()` as public entry point for externally-supplied chunks.
- `Start()` auto-detects `GenerationScheduler` — if present, skips self-scheduling.
- Fully backward-compatible: without a scheduler, behaviour is identical to before.

**ToadGanGenerator changes:**
- Exposed `TileMap` property (read-only accessor to the loaded `itos` vocabulary)
  so the scheduler can build fallback chunks using correct tile IDs.

**PerformanceMonitor changes:**
- Auto-discovers `GenerationScheduler` and displays a new "GENERATION SCHEDULER"
  HUD section when active: buffered chunks, safe prefetch target, EMA timings,
  fallback status, and total ML/fallback chunk counts.
- CSV log extended with 6 scheduler columns (`Sched_Buffered`, `Sched_SafePrefetch`,
  `Sched_EmaInfer`, `Sched_EmaBuild`, `Sched_Fallback`, `Sched_FallbackTotal`).

**Impact**

- Generation is triggered based on *measured* ML cost rather than a fixed tile
  count, automatically adapting to different hardware / backend combinations.
- The chunk buffer absorbs speed bursts (dash) that previously could outrun the
  single-chunk reactive system.
- Fallback chunks guarantee the player always has ground, eliminating the worst
  failure mode (void fall) entirely.
- All scheduling parameters are exposed in a ScriptableObject for A/B testing
  different configurations.
- No change to the ONNX inference pipeline itself — the scheduler is purely a
  wrapper around *when* `Generate()` is called, not *how*.

**vs hand-authored levels:**

Hand-authored levels are pre-loaded assets with zero generation cost — the entire
level exists before the player moves.  ML-generated levels pay an ongoing
inference cost per chunk.  This scheduler specifically bridges that gap by:

1. Making the *timing* of that cost invisible to the player (prefetch).
2. Absorbing *variance* in that cost (EMA + buffer).
3. Providing a *zero-cost fallback* when the model can't keep up.
4. Logging all cost metrics so developers can quantify the overhead vs static levels.

---

## Future Work / Candidates

| ID | Area | Idea | Priority |
|---|---|---|---|
| F-01 | `LevelInstantiator` | Tile GameObject pooling — reuse destroyed tiles instead of Instantiate/Destroy | High |
| F-02 | `LevelInstantiator` | Batch `CleanupOldTiles` destructions (max N per frame) to smooth long cleanup frames | Medium |
| F-03 | `ToadGanGenerator` | Reuse pre-allocated `float[]` noise buffers across `Generate()` calls | Low |
| F-04 | `EnemyPatrol` | Use `Physics2D.RaycastNonAlloc` buffer; currently raycasts return structs (already zero-alloc in Unity 2021+, verify) | Low |
| F-05 | General | Strip `Debug.Log` calls from hot paths in release builds using `[Conditional("UNITY_EDITOR")]` or a custom logger | Medium |
| F-06 | `ToadGanGenerator` | Investigate Unity Sentis async scheduling to move ONNX inference off the main thread — would allow `GenerationScheduler.maxConcurrentGenerations > 1` to truly overlap | High |
| F-07 | `GenerationScheduler` | Track player velocity (not just max speed) for tighter adaptive prefetch when the player is slow-walking | Low |
| F-08 | `GenerationScheduler` | Visual/audio cue when fallback chunk is entered (e.g. brief tile tint or subtle audio indicator) | Low |
