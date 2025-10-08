using System.Collections.Generic;
using UnityEngine;

public class LevelGenerator : MonoBehaviour
{
    [Header("Tile Prefabs")]
    public GameObject groundPrefab;   // '#'
    public GameObject brickPrefab;    // 'B'
    public GameObject questionPrefab; // '?'
    public GameObject coinPrefab;     // 'o'
    public GameObject pipePrefab;     // 'p' and 'P' (stacked vertically)
    public GameObject cannonPrefab;   // 'c'
    public GameObject springPrefab;   // 'y'/'Y'
    public GameObject enemyPrefab;    // 'e','g','k','K','t','l','V','h'
    public GameObject otherPrefab;    // fallback

    [Header("Layout")] 
    public float tileSize = 1f;
    public Vector2 origin = Vector2.zero;
    public bool invertY = true; // text first line is top

    public void BuildLevel(string[] lines)
    {
        if (lines == null || lines.Length == 0) return;

        // Clear previous children
        var toDestroy = new List<GameObject>();
        foreach (Transform child in transform) toDestroy.Add(child.gameObject);
        foreach (var go in toDestroy) Destroy(go);

        int height = lines.Length;
        for (int row = 0; row < height; row++)
        {
            int y = invertY ? (height - 1 - row) : row;
            string ln = lines[row];
            for (int x = 0; x < ln.Length; x++)
            {
                char ch = ln[x];
                var prefab = PrefabForChar(ch);
                if (prefab == null) continue; // sky
                Vector3 pos = new Vector3(origin.x + x * tileSize, origin.y + y * tileSize, 0f);
                Instantiate(prefab, pos, Quaternion.identity, this.transform);
            }
        }
    }

    GameObject PrefabForChar(char ch)
    {
        switch (ch)
        {
            case '#': return groundPrefab;
            case 'B': return brickPrefab;
            case '?': return questionPrefab;
            case 'o': return coinPrefab;
            case 'p':
            case 'P': return pipePrefab;
            case 'c': return cannonPrefab;
            case 'y':
            case 'Y': return springPrefab;
            case 'e':
            case 'g':
            case 'k':
            case 'K':
            case 't':
            case 'l':
            case 'V':
            case 'h': return enemyPrefab;
            case '-':
            case ' ': return null; // sky
            default: return otherPrefab;
        }
    }
}



