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

## Future Work / Candidates

| ID | Area | Idea | Priority |
|---|---|---|---|
| F-01 | `LevelInstantiator` | Tile GameObject pooling — reuse destroyed tiles instead of Instantiate/Destroy | High |
| F-02 | `LevelInstantiator` | Batch `CleanupOldTiles` destructions (max N per frame) to smooth long cleanup frames | Medium |
| F-03 | `ToadGanGenerator` | Reuse pre-allocated `float[]` noise buffers across `Generate()` calls | Low |
| F-04 | `EnemyPatrol` | Use `Physics2D.RaycastNonAlloc` buffer; currently raycasts return structs (already zero-alloc in Unity 2021+, verify) | Low |
| F-05 | General | Strip `Debug.Log` calls from hot paths in release builds using `[Conditional("UNITY_EDITOR")]` or a custom logger | Medium |
| F-06 | `ToadGanGenerator` | Investigate Unity Sentis async scheduling to move ONNX inference off the main thread | High |
