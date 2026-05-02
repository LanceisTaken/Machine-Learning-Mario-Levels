using UnityEngine;

/// <summary>
/// Continuous runtime safety net that detects when an entity (enemy, power-up, etc.)
/// has merged into ground/pipe geometry and teleports it to sit on the ground surface.
///
/// The CompositeCollider2D merges tile colliders into a hollow shell — overlap queries
/// return null when an entity is fully inside. This script works around that by
/// raycasting downward from above the entity to find the true ground surface, then
/// comparing it to the entity's bottom Y to detect embedding.
///
/// Also keeps a secondary overlap check for partially-embedded edge-clipping cases.
/// </summary>
[RequireComponent(typeof(Collider2D))]
public class EntityUnstuck : MonoBehaviour
{
    [Tooltip("Layers treated as ground/walls for overlap detection.")]
    public LayerMask groundLayer;

    [Tooltip("Seconds to skip checks after spawn (just enough for physics init).")]
    public float gracePeriod = 0.15f;

    [Tooltip("Run the check every N FixedUpdate calls after the early-life window.")]
    public int checkInterval = 5;

    [Tooltip("Seconds to pause checks after a successful correction to avoid jitter.")]
    public float cooldownAfterFix = 0.5f;

    [Tooltip("Maximum upward teleport distance (world units). Safety cap against huge jumps.")]
    public float maxTeleportUp = 3f;

    [Tooltip("How far above the entity to start the surface-finding raycast.")]
    public float probeHeight = 5f;

    [Tooltip("Destroy the entity after this many lifetime corrections (stuck in a loop).")]
    public int maxLifetimeFixes = 10;

    [Tooltip("Duration (seconds) after spawn during which checks run every FixedUpdate frame.")]
    public float earlyLifeDuration = 1f;

    private Collider2D _col;
    private Rigidbody2D _rb;
    private float _spawnTime;
    private float _cooldownUntil;
    private int _frameCounter;
    private int _totalFixes;
    private const int MaxLoggedFixes = 3;

    // Cached collider shape info
    private bool _isCircle;
    private CircleCollider2D _circle;
    private BoxCollider2D _box;

    /// <summary>Minimum embed depth (world units) to count as stuck.
    /// Prevents false positives from normal surface contact.</summary>
    private const float EmbedThreshold = 0.05f;

    /// <summary>Shrink factor for the secondary overlap query.</summary>
    private const float OverlapShrink = 0.8f;

    private void Awake()
    {
        _col = GetComponent<Collider2D>();
        _rb = GetComponent<Rigidbody2D>();
        _spawnTime = Time.time;

        _circle = _col as CircleCollider2D;
        _box = _col as BoxCollider2D;
        _isCircle = _circle != null;
    }

    private void FixedUpdate()
    {
        if (_col == null) return;

        float age = Time.time - _spawnTime;

        if (age < gracePeriod) return;
        if (Time.time < _cooldownUntil) return;

        bool earlyLife = age < earlyLifeDuration;
        if (!earlyLife)
        {
            _frameCounter++;
            if (_frameCounter < checkInterval) return;
            _frameCounter = 0;
        }

        // --- Primary detection: raycast from above to find the true ground surface ---
        Bounds bounds = _col.bounds;
        float entityBottomY = bounds.min.y;
        float entityCenterX = bounds.center.x;
        float halfHeight = bounds.extents.y;

        Vector2 rayOrigin = new Vector2(entityCenterX, entityBottomY + probeHeight);
        RaycastHit2D hit = Physics2D.Raycast(rayOrigin, Vector2.down, probeHeight + 1f, groundLayer);

        if (hit.collider != null)
        {
            float surfaceY = hit.point.y;
            float entityTopY = bounds.max.y;

            if (surfaceY <= entityTopY + 0.02f)
            {
                // Surface is at or below entity top → standard embedding check
                float embedDepth = surfaceY - entityBottomY;

                if (embedDepth > EmbedThreshold)
                {
                    float newY = surfaceY + halfHeight + EmbedThreshold;
                    if (newY - transform.position.y > maxTeleportUp)
                        newY = transform.position.y + maxTeleportUp;

                    ApplyFix(newY, embedDepth);
                    return;
                }
            }
            else
            {
                // Surface is above entity top → could be a ceiling or the
                // entity is enclosed inside a hollow structure. Cast short
                // horizontal rays to distinguish the two cases.
                float sideCheckDist = 1.5f;
                Vector2 center = bounds.center;
                bool wallLeft  = Physics2D.Raycast(center, Vector2.left,  sideCheckDist, groundLayer).collider != null;
                bool wallRight = Physics2D.Raycast(center, Vector2.right, sideCheckDist, groundLayer).collider != null;

                if (wallLeft && wallRight)
                {
                    // Walled in on both sides → genuinely inside hollow geometry
                    float newY = surfaceY + halfHeight + EmbedThreshold;
                    if (newY - transform.position.y > maxTeleportUp)
                        newY = transform.position.y + maxTeleportUp;

                    ApplyFix(newY, surfaceY - entityBottomY);
                    return;
                }
            }
        }

        // --- Secondary detection: overlap check for partial edge-clipping ---
        if (IsOverlappingGround())
        {
            // Fallback: step up iteratively until the overlap clears
            Vector3 pos = transform.position;
            float totalShift = 0f;
            float step = 0.25f;

            while (totalShift < maxTeleportUp)
            {
                pos.y += step;
                totalShift += step;
                // Re-check using bounds at the candidate position
                Vector2 testCenter = new Vector2(pos.x + _col.offset.x, pos.y + _col.offset.y);
                Vector2 testSize = (Vector2)bounds.size * OverlapShrink;
                if (Physics2D.OverlapBox(testCenter, testSize, 0f, groundLayer) == null)
                    break;
            }

            if (totalShift > 0f)
                ApplyFix(pos.y, totalShift);
        }
    }

    private void ApplyFix(float newY, float shiftAmount)
    {
        Vector3 oldPos = transform.position;
        Vector3 newPos = new Vector3(oldPos.x, newY, 0f);

        transform.position = newPos;
        if (_rb != null)
        {
            _rb.position = new Vector2(newPos.x, newPos.y);
            _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, 0f);
        }

        _cooldownUntil = Time.time + cooldownAfterFix;
        _totalFixes++;

        if (_totalFixes <= MaxLoggedFixes)
        {
            Debug.Log($"[EntityUnstuck] {name} was stuck at {oldPos}, moved to {newPos} " +
                      $"(+{shiftAmount:F2}u, fix #{_totalFixes})");
        }

        if (_totalFixes >= maxLifetimeFixes)
        {
            Debug.LogWarning($"[EntityUnstuck] {name} exceeded {maxLifetimeFixes} fixes — " +
                             "destroying to prevent infinite loop.");
            Destroy(gameObject);
        }
    }

    /// <summary>
    /// Secondary overlap check for partial embedding / edge clipping.
    /// Uses the correct query shape for the entity's collider type.
    /// </summary>
    private bool IsOverlappingGround()
    {
        if (_isCircle)
        {
            Vector2 center = (Vector2)transform.position + _circle.offset;
            float radius = _circle.radius * Mathf.Max(transform.localScale.x, transform.localScale.y);
            return Physics2D.OverlapCircle(center, radius * OverlapShrink, groundLayer) != null;
        }

        if (_box != null)
        {
            Vector2 center = (Vector2)transform.position + _box.offset;
            Vector2 size = _box.size * OverlapShrink;
            return Physics2D.OverlapBox(center, size, 0f, groundLayer) != null;
        }

        Vector2 fbCenter = (Vector2)transform.position + _col.offset;
        Vector2 fbSize = (Vector2)_col.bounds.size * OverlapShrink;
        return Physics2D.OverlapBox(fbCenter, fbSize, 0f, groundLayer) != null;
    }
}
