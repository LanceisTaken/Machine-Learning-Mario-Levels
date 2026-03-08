using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Parses the JSON payload returned by the TOAD-GAN Flask server and
/// instantiates tile prefabs to build the level in-scene.
///
/// Setup
/// -----
/// 1. Attach this component to an empty "LevelRoot" GameObject.
/// 2. Assign a prefab for each tile character you care about in the
///    <see cref="tilePrefabs"/> list.
/// 3. Drop your <see cref="ApiClient"/> component into the matching slot.
/// 4. Hit Play – the level will be fetched and built automatically.
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

    [Header("API")]
    [Tooltip("The ApiClient that will fetch the level data.")]
    public ApiClient apiClient;

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
    [Tooltip("How close (in tiles) the player can get to the level edge before generating more.")]
    public float generateAheadDistance = 10f;

    [Tooltip("How far behind the player (in units) before tiles get destroyed.")]
    public float destroyBehindDistance = 50f;

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
        if (GetComponent<CompositeCollider2D>() == null)
        {
            gameObject.AddComponent<CompositeCollider2D>();
        }

        // Put this root object on the Ground layer so the merged
        // CompositeCollider2D is detected by player ground-checks
        // and enemy wall raycasts.
        // If groundLayer wasn't assigned in the Inspector, try to
        // auto-detect it from the player's PlayerController.
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
            // LayerMask.value is a bitmask; convert to the single layer index
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

        if (apiClient == null)
        {
            Debug.LogError("[LevelInstantiator] No ApiClient assigned! Assign one in the Inspector.");
            return;
        }

        apiClient.OnLevelReceived += HandleLevelJson;
        apiClient.OnError        += err => Debug.LogError("[LevelInstantiator] " + err);

        // Request the first level immediately on Play
        apiClient.RequestLevel();
    }

    private void Update()
    {
        if (player == null || apiClient == null) return;

        CleanupOldTiles();

        if (_isGenerating) return;

        float levelEdge = _nextChunkX;
        float playerX = player.transform.position.x;

        if (playerX + (generateAheadDistance * tileSize) >= levelEdge)
        {
            _isGenerating = true;
            apiClient.RequestLevel();
        }
    }

    private void OnDestroy()
    {
        if (apiClient != null)
            apiClient.OnLevelReceived -= HandleLevelJson;
    }

    // ── Level building ────────────────────────────────────────────────────

    /// <summary>Parse the JSON string and build the level.</summary>
    private void HandleLevelJson(string json)
    {
        LevelPayload payload;
        try
        {
            payload = JsonConvert.DeserializeObject<LevelPayload>(json);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[LevelInstantiator] Failed to parse JSON: {ex.Message}");
            _isGenerating = false;
            return;
        }

        if (payload?.tile_ids == null || payload.tile_ids.Length == 0)
        {
            Debug.LogError("[LevelInstantiator] Payload is empty or malformed.");
            _isGenerating = false;
            return;
        }

        var idToChar = new Dictionary<int, char>();
        if (payload.tile_map != null)
        {
            foreach (var kvp in payload.tile_map)
            {
                if (int.TryParse(kvp.Key, out int id) && kvp.Value.Length > 0)
                    idToChar[id] = kvp.Value[0];
            }
        }

        Debug.Log($"[LevelInstantiator] Appending chunk at X={_nextChunkX}: {payload.height}h x {payload.width}w");
        BuildChunk(payload.tile_ids, idToChar, payload.height, payload.width);

        if (!_playerSpawned && player != null)
        {
            SpawnPlayer(payload.tile_ids, idToChar, payload.height, payload.width);
            _playerSpawned = true;
        }
        _isGenerating = false;
    }

    /// <summary>Destroy all previously spawned tiles.</summary>
    public void ClearLevel()
    {
        foreach (Transform child in transform)
            Destroy(child.gameObject);
    }

    private void BuildChunk(
        int[][] tileIds,
        Dictionary<int, char> idToChar,
        int height,
        int width)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int id = tileIds[row][col];

                if (!idToChar.TryGetValue(id, out char ch)) continue;
                if (ch == '-' || ch == '\n') continue;

                if (!_prefabMap.TryGetValue(ch, out GameObject prefab))
                    continue;

                float x = _nextChunkX + (col * tileSize);
                float y = (height - 1 - row) * tileSize;

                GameObject tile = Instantiate(prefab,
                    new Vector3(x, y, 0f),
                    Quaternion.identity,
                    transform);

                tile.name = $"Tile_{ch}_{col}_{row}";

                // Merge static terrain colliders into the parent
                // CompositeCollider2D so adjacent tiles form one seamless
                // surface — eliminates invisible "dead spot" seams.
                // Skip dynamic objects (enemies, question blocks, coins, etc.)
                // that need their own independent colliders.
                bool isStaticTerrain = ch == '#' || ch == 'S' ||   // solid / brick
                                       ch == '<' || ch == '>' ||   // pipe top
                                       ch == '[' || ch == ']' ||   // pipe body
                                       ch == '§';                  // compound pipe
                if (isStaticTerrain)
                {
                    BoxCollider2D bc = tile.GetComponent<BoxCollider2D>();
                    if (bc != null)
                        bc.compositeOperation = Collider2D.CompositeOperation.Merge;
                }
            }
        }

        // Move the offset forward for the next chunk
        _nextChunkX += width * tileSize;

        Debug.Log($"[LevelInstantiator] Chunk built. Next chunk starts at X={_nextChunkX}");
    }

    private void SpawnPlayer(int[][] tileIds, Dictionary<int, char> idToChar, int height, int width)
    {
        // Scan the first few columns to find a safe spawn point.
        // A safe spawn = a solid ground tile with at least 2 empty rows above it
        // (so the player has room to stand without overlapping any collider).
        for (int col = 0; col < Mathf.Min(5, width); col++)
        {
            // Scan top-down to find the ground surface
            for (int row = 0; row < height; row++)
            {
                if (IsEmpty(tileIds, idToChar, row, col)) continue;

                // row is the first solid tile from the top (ground surface).
                // Verify the two rows above it are clear (player needs headroom).
                int spawnRow = row - 1;  // one tile above the surface
                if (spawnRow < 0) continue;

                bool aboveClear = IsEmpty(tileIds, idToChar, spawnRow, col);
                bool twoClear   = spawnRow - 1 < 0 || IsEmpty(tileIds, idToChar, spawnRow - 1, col);

                if (!aboveClear || !twoClear)
                {
                    // Not enough headroom in this column, try the next one
                    break;
                }

                // Safe spawn position: one tile above the ground surface
                float x = col * tileSize;
                float y = (height - 1 - spawnRow) * tileSize + 0.05f;

                // Use Rigidbody2D.position for teleporting — setting only
                // transform.position can desync from the physics body for one
                // frame, causing the "stuck in collider" bug.
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

        // Fallback: top-left corner
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

    // ── JSON data structures ──────────────────────────────────────────────
    // These mirror the payload produced by generate.py / server.py.
    // Newtonsoft.Json handles int[][] and Dictionary natively — no custom
    // wrapper classes needed.

    private void CleanupOldTiles()
    {
        float cutoff = player.transform.position.x - destroyBehindDistance;
        foreach (Transform child in transform)
        {
            if (child.position.x < cutoff)
                Destroy(child.gameObject);
        }
    }

    private class LevelPayload
    {
        public int                       height;
        public int                       width;
        public int[][]                   tile_ids;
        // tile_map comes from the server as {"0":"-", "1":"#", ...}
        public Dictionary<string, string> tile_map;
    }
}
