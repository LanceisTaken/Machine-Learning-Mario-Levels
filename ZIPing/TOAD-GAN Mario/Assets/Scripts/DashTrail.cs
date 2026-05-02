using System.Collections;
using UnityEngine;

/// <summary>
/// Attach to the Player GameObject.
/// Call StartTrail() / StopTrail() from PlayerController during a dash.
/// Continuously spawns semi-transparent ghost copies of the player sprite.
/// </summary>
[RequireComponent(typeof(SpriteRenderer))]
public class DashTrail : MonoBehaviour
{
    [Header("Trail Settings")]
    [Tooltip("How many seconds between ghost spawns.")]
    public float spawnInterval = 0.04f;

    [Tooltip("Starting alpha of each ghost.")]
    public float ghostAlpha = 0.5f;

    [Tooltip("How long each ghost takes to fully fade.")]
    public float fadeDuration = 0.18f;

    [Tooltip("Tint colour for the ghost sprites.")]
    public Color ghostTint = new Color(0.5f, 0.8f, 1f, 1f); // light-blue tint

    private SpriteRenderer _sr;
    private Coroutine _trailCoroutine;

    private void Awake()
    {
        _sr = GetComponent<SpriteRenderer>();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    public void StartTrail()
    {
        if (_trailCoroutine != null)
            StopCoroutine(_trailCoroutine);
        _trailCoroutine = StartCoroutine(SpawnGhosts());
    }

    public void StopTrail()
    {
        if (_trailCoroutine != null)
        {
            StopCoroutine(_trailCoroutine);
            _trailCoroutine = null;
        }
    }

    // ── Ghost spawning ─────────────────────────────────────────────────────

    private IEnumerator SpawnGhosts()
    {
        while (true)
        {
            SpawnGhost();
            yield return new WaitForSeconds(spawnInterval);
        }
    }

    private void SpawnGhost()
    {
        // Create a new sprite-only GameObject at the current player position
        GameObject ghost = new GameObject("DashGhost");
        ghost.transform.position   = transform.position;
        ghost.transform.localScale = transform.localScale;
        ghost.transform.rotation   = transform.rotation;

        SpriteRenderer ghostSr = ghost.AddComponent<SpriteRenderer>();
        ghostSr.sprite         = _sr.sprite;
        ghostSr.flipX          = _sr.flipX;
        ghostSr.sortingLayerID = _sr.sortingLayerID;
        ghostSr.sortingOrder   = _sr.sortingOrder - 1; // render behind player

        Color startColor = new Color(ghostTint.r, ghostTint.g, ghostTint.b, ghostAlpha);
        ghostSr.color = startColor;

        StartCoroutine(FadeGhost(ghostSr, ghost));
    }

    private IEnumerator FadeGhost(SpriteRenderer ghostSr, GameObject ghost)
    {
        float elapsed = 0f;
        Color startColor = ghostSr.color;

        while (elapsed < fadeDuration)
        {
            float alpha = Mathf.Lerp(startColor.a, 0f, elapsed / fadeDuration);
            ghostSr.color = new Color(startColor.r, startColor.g, startColor.b, alpha);
            elapsed += Time.deltaTime;
            yield return null;
        }

        Destroy(ghost);
    }
}
