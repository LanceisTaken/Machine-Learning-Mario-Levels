using System.Collections.Generic;
using UnityEngine;

public class LevelGenerator : MonoBehaviour
{
    [Header("Tile Sprites")]
    public Sprite groundSprite;   // '#'
    public Sprite brickSprite;    // 'B'
    public Sprite questionSprite; // '?'
    public Sprite coinSprite;     // 'o'
    public Sprite pipeSprite;     // 'p' and 'P'
    public Sprite cannonSprite;   // 'c'
    public Sprite springSprite;   // 'y'/'Y'
    public Sprite enemySprite;    // 'e','g','k','K','t','l','V','h'
    public Sprite otherSprite;    // fallback

    [Header("Layout")] 
    public float tileSize = 1f;
    public Vector2 origin = Vector2.zero;
    public bool invertY = true; // text first line is top
    public int sortingOrder = 0; // SpriteRenderer sorting order

    public void BuildLevel(string[] lines)
    {
        if (lines == null || lines.Length == 0) return;

        // Clear previous children
        var toDestroy = new List<GameObject>();
        foreach (Transform child in transform) toDestroy.Add(child.gameObject);
        foreach (var go in toDestroy) Destroy(go);

        int height = lines.Length;
        int placed = 0;
        for (int row = 0; row < height; row++)
        {
            int y = invertY ? (height - 1 - row) : row;
            string ln = lines[row];
            for (int x = 0; x < ln.Length; x++)
            {
                char ch = ln[x];
                var sprite = SpriteForChar(ch);
                if (sprite == null) continue; // sky

                var go = new GameObject($"tile_{x}_{y}");
                go.transform.SetParent(this.transform, false);
                go.transform.position = new Vector3(origin.x + x * tileSize, origin.y + y * tileSize, 0f);
                var sr = go.AddComponent<SpriteRenderer>();
                sr.sprite = sprite;
                sr.sortingOrder = sortingOrder;
                placed++;
            }
        }
        Debug.Log($"LevelGenerator: placed {placed} tiles, rows={height}, cols={(height>0?lines[0].Length:0)}");
    }

    Sprite SpriteForChar(char ch)
    {
        switch (ch)
        {
            case '#': return groundSprite;
            case 'B': return brickSprite;
            case '?': return questionSprite;
            case 'o': return coinSprite;
            case 'p':
            case 'P': return pipeSprite;
            case 'c': return cannonSprite;
            case 'y':
            case 'Y': return springSprite;
            case 'e':
            case 'g':
            case 'k':
            case 'K':
            case 't':
            case 'l':
            case 'V':
            case 'h': return enemySprite;
            case '-':
            case ' ': return null; // sky
            default: return otherSprite;
        }
    }
}



