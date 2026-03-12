using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

/// <summary>
/// Consumes tile data from <see cref="ToadGanGenerator"/> (local Sentis
/// inference) and instantiates tile prefabs to build the level in-scene.
///
/// Setup
/// -----
/// 1. Attach this component to an empty "LevelRoot" GameObject.
/// 2. Assign a prefab for each tile character you care about in the
///    <see cref="tilePrefabs"/> list.
/// 3. Drop your <see cref="ToadGanGenerator"/> component into the
///    <see cref="generator"/> slot.
/// 4. Hit Play – a level will be generated and built automatically.
///
/// Tile map (TOAD-GAN default vocab)
/// ----------------------------------
///   #  Solid / unbreakable block
///   S  Brick / breakable block
///   ?  Lucky (question-mark) block
///   Q  Used question-mark block
///   &lt;  Pipe top-left
///   &gt;  Pipe top-right
///   [  Pipe body-left
///   ]  Pipe body-right
///   §  Compound pipe token (handled as pipe body)
///   o  Coin
///   E  Enemy spawn point
///   g  Goomba spawn point
///   -  Empty / air  (skip – nothing is instantiated)
///   (everything else is ignored silently)
/// </summary>

public class LevelInstantiator : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────

    [Header("Generator")]
    [Tooltip("The ToadGanGenerator that runs local ONNX inference.")]
    public ToadGanGenerator generator;

    [Header("Tile Size")]
    [Tooltip("World-space size (in Unity units) of one tile. Default = 1.")]
    public float tileSize = 1f;

    [Header("Physics")]
    [Tooltip("Set this to your Ground layer so the CompositeCollider2D is on the correct layer for player ground-checks and enemy wall-checks.")]
    public LayerMask groundLayer;

    [Header("Tile Prefab Mapping")]
    [Tooltip("Map a single character to its prefab. Leave 'air' characters unmapped.")]
    public List<TilePrefabEntry> tilePrefabs = new();

    [Header("Player")]
    [Tooltip("The player prefab or existing player in the scene.")]
    public GameObject player;

    [Header("Infinite Generation")]
    [Tooltip("How close (in tiles) the player can get to the level edge before generating more. Increase for a larger safety buffer before the chunk arrives.")]
    public float generateAheadDistance = 20f;

    [Tooltip("How far behind the player (in units) before tiles get destroyed.")]
    public float destroyBehindDistance = 50f;

    [Header("Performance")]
    [Tooltip("Maximum tile GameObjects instantiated per frame during chunk building. Lower = smoother framerate during build; higher = chunk appears faster.")]
    public int tilesPerFrameBudget = 25;

    /// <summary>
    /// Raised after each chunk is fully instantiated and the CompositeCollider2D
    /// geometry has been rebuilt.
    /// Arguments: wall-clock build time in milliseconds, number of tiles spawned.
    /// </summary>
    public event Action<float, int> OnChunkBuilt;

    /// <summary>Tracks the X offset (in world units) where the next chunk should start.</summary>
    private float _nextChunkX = 0f;

    /// <summary>Prevents overlapping generation requests.</summary>
    private bool _isGenerating = false;

    /// <summary>Tracks whether the player has been spawned in the level.</summary>
    private bool _playerSpawned = false;

    // ── Serialisable helper ───────────────────────────────────────────────

    [Serializable]
    public class TilePrefabEntry
    {
        [Tooltip("The single character this prefab represents (e.g. '#', '?').")]
        public string character = "#";
        public GameObject prefab;
    }

    // ── Runtime state ─────────────────────────────────────────────────────

    /// <summary>Lookup built from <see cref="tilePrefabs"/> at Start.</summary>
    private Dictionary<char, GameObject> _prefabMap;

    /// <summary>
    /// All currently-alive tile Transforms, maintained so CleanupOldTiles
    /// never has to enumerate the Transform hierarchy (which allocates an
    /// IEnumerator every call).
    /// </summary>
    private readonly List<Transform> _activeTiles = new List<Transform>();

    /// <summary>
    /// World-space X position of each tile in <see cref="_activeTiles"/>,
    /// cached at spawn time.  Static tiles never move, so this never needs
    /// to be refreshed — it lets CleanupOldTiles compare plain floats
    /// instead of calling Transform.position (a native C++ property) for
    /// every tile every frame.  Accessing transform positions while the
    /// physics job system is running (i.e. whenever the player is moving)
    /// causes a job-sync stall; this eliminates that entirely.
    /// </summary>
    private readonly List<float> _activeTileXs = new List<float>();

    /// <summary>
    /// Player X position at the last CleanupOldTiles pass.  The loop only
    /// runs again once the player has moved at least
    /// <see cref="CleanupMoveThreshold"/> units, skipping the iteration
    /// entirely on frames where nothing could have crossed the cutoff.
    /// </summary>
    private float _lastCleanupPlayerX = float.MinValue;
    private const float CleanupMoveThreshold = 0.25f;

    /// <summary>Cached reference to the CompositeCollider2D on this object.</summary>
    private CompositeCollider2D _composite;

    /// <summary>
    /// Cached Rigidbody2D of the player.  Reading Rigidbody2D.position is safe
    /// from the main thread without stalling the physics job, unlike reading
    /// Transform.position while a Rigidbody2D is active.
    /// </summary>
    private Rigidbody2D _playerRb;

    // ── Unity lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        // Build fast character → prefab lookup
        _prefabMap = new Dictionary<char, GameObject>();
        foreach (var entry in tilePrefabs)
        {
            if (entry.prefab == null || entry.character.Length == 0) continue;
            _prefabMap[entry.character[0]] = entry.prefab;
        }

        // ── CompositeCollider2D setup ──────────────────────────────────
        // Add a static Rigidbody2D + CompositeCollider2D to this root
        // object.  All child tile BoxCollider2Ds set to "Merge" mode will
        // be combined into one seamless collider surface, eliminating the
        // tile-edge seams that catch the player's box corners.
        if (GetComponent<Rigidbody2D>() == null)
        {
            var rb = gameObject.AddComponent<Rigidbody2D>();
            rb.bodyType = RigidbodyType2D.Static;
        }

        _composite = GetComponent<CompositeCollider2D>();
        if (_composite == null)
            _composite = gameObject.AddComponent<CompositeCollider2D>();

        // Use Manual generation so each tile's compositeOperation assignment
        // does NOT trigger an immediate geometry rebuild.  We call
        // GenerateGeometry() once per chunk after all tiles are placed,
        // replacing ~350 incremental rebuilds with a single one.
        _composite.generationType = CompositeCollider2D.GenerationType.Manual;

        // Put this root object on the Ground layer so the merged
        // CompositeCollider2D is detected by player ground-checks
        // and enemy wall raycasts.
        if (groundLayer.value == 0 && player != null)
        {
            var pc = player.GetComponent<PlayerController>();
            if (pc != null && pc.groundLayer.value != 0)
            {
                groundLayer = pc.groundLayer;
                Debug.Log($"[LevelInstantiator] Auto-detected groundLayer from PlayerController: {groundLayer.value}");
            }
        }

        if (groundLayer.value != 0)
        {
            int layerIndex = 0;
            int bits = groundLayer.value;
            while (bits > 1) { bits >>= 1; layerIndex++; }
            gameObject.layer = layerIndex;
            Debug.Log($"[LevelInstantiator] Root object set to layer {layerIndex} ({LayerMask.LayerToName(layerIndex)})");
        }
        else
        {
            Debug.LogWarning("[LevelInstantiator] groundLayer is not set! " +
                "Player ground-checks and enemy wall-checks may not work. " +
                "Assign the Ground layer in the Inspector.");
        }

        if (generator == null)
        {
            Debug.LogError("[LevelInstantiator] No ToadGanGenerator assigned! Assign one in the Inspector.");
            return;
        }

        generator.OnLevelGenerated += HandleGeneratedLevel;
        generator.OnError          += err => Debug.LogError("[LevelInstantiator] " + err);

        if (player != null)
            _playerRb = player.GetComponent<Rigidbody2D>();

        // Request the first level immediately on Play
        generator.Generate();
    }

    private void Update()
    {
        if (player == null || generator == null) return;

        CleanupOldTiles();

        if (_isGenerating) return;

        float levelEdge = _nextChunkX;
        float playerX = _playerRb != null ? _playerRb.position.x : player.transform.position.x;

        if (playerX + (generateAheadDistance * tileSize) >= levelEdge)
        {
            _isGenerating = true;
            generator.Generate();
        }
    }

    private void OnDestroy()
    {
        if (generator != null)
            generator.OnLevelGenerated -= HandleGeneratedLevel;
    }

    // ── Level building ────────────────────────────────────────────────────

    /// <summary>
    /// Receives tile data directly from <see cref="ToadGanGenerator"/> and
    /// starts a coroutine to build the next level chunk over multiple frames.
    /// </summary>
    private void HandleGeneratedLevel(
        int[][] tileIds,
        Dictionary<string, string> tileMap,
        int height,
        int width)
    {
        if (tileIds == null || tileIds.Length == 0)
        {
            Debug.LogError("[LevelInstantiator] Received empty tile data.");
            _isGenerating = false;
            return;
        }

        var idToChar = new Dictionary<int, char>();
        if (tileMap != null)
        {
            foreach (var kvp in tileMap)
            {
                if (int.TryParse(kvp.Key, out int id) && kvp.Value.Length > 0)
                    idToChar[id] = kvp.Value[0];
            }
        }

        bool needSpawn = !_playerSpawned && player != null;
        if (needSpawn) _playerSpawned = true; // Mark now to prevent a double-spawn if another chunk arrives quickly

        Debug.Log($"[LevelInstantiator] Appending chunk at X={_nextChunkX}: {height}h x {width}w");
        StartCoroutine(BuildChunkCoroutine(tileIds, idToChar, height, width, needSpawn));
        // _isGenerating stays true; the coroutine clears it when done
    }

    /// <summary>Destroy all previously spawned tiles and clear the tracking lists.</summary>
    public void ClearLevel()
    {
        foreach (var t in _activeTiles)
        {
            if (t != null) Destroy(t.gameObject);
        }
        _activeTiles.Clear();
        _activeTileXs.Clear();
    }

    /// <summary>
    /// Builds a chunk incrementally over multiple frames to avoid a single
    /// large stall.  At the end, calls <see cref="CompositeCollider2D.GenerateGeometry"/>
    /// once rather than letting Unity rebuild the mesh on every tile addition.
    /// </summary>
    private IEnumerator BuildChunkCoroutine(
        int[][] tileIds,
        Dictionary<int, char> idToChar,
        int height,
        int width,
        bool spawnPlayer)
    {
        var stopwatch = Stopwatch.StartNew();
        int tilesSpawned = 0;
        int frameBudgetUsed = 0;

        // Record start X now so _nextChunkX can be advanced atomically at the end
        float chunkStartX = _nextChunkX;

        // Enemy spawns are deferred until after GenerateGeometry() so that the
        // ground colliders exist when the enemies' Rigidbody2Ds first simulate.
        // We store grid row/col so we can scan the tile data for the actual
        // ground surface and position enemies precisely on top of it.
        var pendingEnemies = new List<(GameObject prefab, int row, int col)>();

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int id = tileIds[row][col];

                if (!idToChar.TryGetValue(id, out char ch)) continue;
                if (ch == '-' || ch == '\n') continue;

                if (!_prefabMap.TryGetValue(ch, out GameObject prefab))
                    continue;

                bool isEnemy = ch == 'E' || ch == 'g';
                if (isEnemy)
                {
                    pendingEnemies.Add((prefab, row, col));
                    continue;
                }

                float x = chunkStartX + (col * tileSize);
                float y = (height - 1 - row) * tileSize;

                GameObject tile = Instantiate(prefab,
                    new Vector3(x, y, 0f),
                    Quaternion.identity,
                    transform);

                bool isStaticTerrain = ch == '#' || ch == 'S' ||
                                       ch == '<' || ch == '>' ||
                                       ch == '[' || ch == ']' ||
                                       ch == '§';
                if (isStaticTerrain)
                {
                    BoxCollider2D bc = tile.GetComponent<BoxCollider2D>();
                    if (bc != null)
                        bc.compositeOperation = Collider2D.CompositeOperation.Merge;
                }

                _activeTiles.Add(tile.transform);
                _activeTileXs.Add(x); // cache world-X so CleanupOldTiles never reads Transform.position
                tilesSpawned++;
                frameBudgetUsed++;

                if (frameBudgetUsed >= tilesPerFrameBudget)
                {
                    frameBudgetUsed = 0;
                    yield return null; // spread work across frames
                }
            }
        }

        // Single geometry rebuild now that all tiles are placed.
        // This replaces ~350 incremental rebuilds that Synchronous mode would have done.
        if (_composite != null)
            _composite.GenerateGeometry();

        // Make sure the rebuilt CompositeCollider2D is visible to overlap
        // queries before we spawn enemies.
        Physics2D.SyncTransforms();

        foreach (var (enemyPrefab, enemyRow, enemyCol) in pendingEnemies)
        {
            float x = chunkStartX + (enemyCol * tileSize);
            float y = (height - 1 - enemyRow) * tileSize;

            GameObject enemy = Instantiate(enemyPrefab, new Vector3(x, y, 0f), Quaternion.identity);

            // Compute the final spawn Y before touching the enemy's transform or
            // Rigidbody2D.  Moving transform.position while a Rigidbody2D is active
            // doesn't move the physics body — on the next FixedUpdate it snaps back.
            // Instead we calculate the correct Y here and write it once to both.
            Collider2D enemyCollider = enemy.GetComponent<Collider2D>();
            if (enemyCollider != null && groundLayer.value != 0)
            {
                // Read collider geometry from the component directly — bounds.center
                // can lag after a fresh Instantiate if physics hasn't synced yet.
                Vector2 colOffset;
                Vector2 colSize;
                if (enemyCollider is CircleCollider2D circ)
                {
                    colOffset = circ.offset;
                    colSize   = new Vector2(circ.radius * 2f, circ.radius * 2f);
                }
                else if (enemyCollider is BoxCollider2D box2)
                {
                    colOffset = box2.offset;
                    colSize   = box2.size;
                }
                else
                {
                    colOffset = enemyCollider.offset;
                    colSize   = enemyCollider.bounds.size;
                }

                float finalY = y;
                float step   = tileSize * 0.25f;

                for (int attempts = 0; attempts < 20; attempts++)
                {
                    // Build the query center from our local finalY — no transform
                    // reads, so there is no stale-physics-state risk.
                    Vector2 queryCenter = new Vector2(x + colOffset.x, finalY + colOffset.y);

                    if (Physics2D.OverlapBox(queryCenter, colSize * 0.95f, 0f, groundLayer) == null)
                        break;

                    finalY += step;
                }

                // Write the resolved position to both transform and Rigidbody2D so
                // the physics body starts from the correct location on its first step.
                enemy.transform.position = new Vector3(x, finalY, 0f);
                var rb = enemy.GetComponent<Rigidbody2D>();
                if (rb != null)
                    rb.position = new Vector2(x, finalY);
            }

            _activeTiles.Add(enemy.transform);
            _activeTileXs.Add(x);
            tilesSpawned++;
        }

        _nextChunkX = chunkStartX + width * tileSize;

        if (spawnPlayer)
            SpawnPlayer(tileIds, idToChar, height, width);

        stopwatch.Stop();
        float buildTimeMs = (float)stopwatch.Elapsed.TotalMilliseconds;

        Debug.Log($"[LevelInstantiator] Chunk built ({tilesSpawned} tiles, {buildTimeMs:F1} ms). " +
                  $"Next chunk starts at X={_nextChunkX}");

        OnChunkBuilt?.Invoke(buildTimeMs, tilesSpawned);
        _isGenerating = false;
    }

    private void SpawnPlayer(int[][] tileIds, Dictionary<int, char> idToChar, int height, int width)
    {
        for (int col = 0; col < Mathf.Min(5, width); col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (IsEmpty(tileIds, idToChar, row, col)) continue;

                int spawnRow = row - 1;
                if (spawnRow < 0) continue;

                bool aboveClear = IsEmpty(tileIds, idToChar, spawnRow, col);
                bool twoClear   = spawnRow - 1 < 0 || IsEmpty(tileIds, idToChar, spawnRow - 1, col);

                if (!aboveClear || !twoClear)
                    break;

                float x = col * tileSize;
                float y = (height - 1 - spawnRow) * tileSize + 0.05f;

                var rb = player.GetComponent<Rigidbody2D>();
                if (rb != null)
                {
                    rb.position  = new Vector2(x, y);
                    rb.linearVelocity = Vector2.zero;
                }
                player.transform.position = new Vector3(x, y, 0f);

                Debug.Log($"[LevelInstantiator] Player spawned at ({x}, {y})  " +
                          $"[ground '{GetChar(tileIds, idToChar, row, col)}' at row {row}, col {col}]");
                return;
            }
        }

        player.transform.position = new Vector3(0f, height * tileSize, 0f);
        Debug.LogWarning("[LevelInstantiator] No safe spawn found, using fallback.");
    }

    /// <summary>Returns true if the cell is air / empty / out of bounds.</summary>
    private bool IsEmpty(int[][] tileIds, Dictionary<int, char> idToChar, int row, int col)
    {
        if (row < 0 || row >= tileIds.Length) return true;
        if (col < 0 || col >= tileIds[row].Length) return true;
        int id = tileIds[row][col];
        if (!idToChar.TryGetValue(id, out char ch)) return true;
        return ch == '-' || ch == '\n';
    }

    /// <summary>Returns the tile character at (row, col), or '?' if unmapped.</summary>
    private char GetChar(int[][] tileIds, Dictionary<int, char> idToChar, int row, int col)
    {
        if (row < 0 || row >= tileIds.Length || col < 0 || col >= tileIds[row].Length) return '?';
        return idToChar.TryGetValue(tileIds[row][col], out char ch) ? ch : '?';
    }

    /// <summary>
    /// Destroys tiles that have scrolled far enough behind the player.
    ///
    /// Performance design:
    /// 1. Movement-gated: the loop is skipped entirely unless the player has
    ///    moved at least <see cref="CleanupMoveThreshold"/> units since the
    ///    last pass, which is true on most frames while standing still or
    ///    moving slowly.
    /// 2. No Transform reads: tile world-X values are compared from the
    ///    pre-cached <see cref="_activeTileXs"/> float list.  Reading
    ///    Transform.position for every tile every frame stalls the main
    ///    thread while physics jobs are running (the player's Rigidbody2D
    ///    being active is enough to trigger this), causing a hard FPS drop
    ///    on the frame movement starts.
    /// </summary>
    private void CleanupOldTiles()
    {
        float playerX = _playerRb != null ? _playerRb.position.x : player.transform.position.x;

        // Skip the loop if the player hasn't moved enough since last pass.
        if (Mathf.Abs(playerX - _lastCleanupPlayerX) < CleanupMoveThreshold) return;
        _lastCleanupPlayerX = playerX;

        float cutoff = playerX - destroyBehindDistance;

        // Swap-and-pop: move the last element into the removed slot instead of
        // shifting all elements above i down.  Safe here because (a) we iterate
        // backwards so every element at index > i was already processed without
        // removal, and (b) tile order in the list has no gameplay meaning.
        for (int i = _activeTiles.Count - 1; i >= 0; i--)
        {
            if (_activeTileXs[i] < cutoff)
            {
                if (_activeTiles[i] != null) Destroy(_activeTiles[i].gameObject);

                int last = _activeTiles.Count - 1;
                _activeTiles[i]  = _activeTiles[last];
                _activeTileXs[i] = _activeTileXs[last];
                _activeTiles.RemoveAt(last);
                _activeTileXs.RemoveAt(last);
            }
        }
    }
}
