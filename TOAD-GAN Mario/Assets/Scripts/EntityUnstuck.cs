using UnityEngine;

/// <summary>
/// Continuous runtime safety net that detects when an entity has merged into
/// ground/pipe geometry and teleports it upward to clear space.
///
/// Strategy: instead of OverlapBox (which silently fails inside a hollow
/// CompositeCollider2D shell), we cast a ray DOWNWARD from well above the
/// entity to find the true ground surface, then compare that surface Y to the
/// entity's feet. If feet are below the surface, the entity is embedded.
///
/// A secondary OverlapBox/OverlapCircle check catches partial-embedding at
/// the collider edge before the entity is fully inside the hollow.
/// </summary>
[RequireComponent(typeof(Collider2D))]
public class EntityUnstuck : MonoBehaviour
{
    [Tooltip("Layers treated as ground/walls for overlap detection.")]
    public LayerMask groundLayer;

    [Tooltip("Seconds after spawn to skip checks — lets spawn-correction settle.")]
    public float gracePeriod = 0.15f;

    [Tooltip("How high above the entity (units) to start the surface-finding ray.")]
    public float probeHeight = 5f;

    [Tooltip("Units to step upward each iteration when nudging out of ground.")]
    public float stepSize = 0.25f;

    [Tooltip("Maximum cumulative upward shift (world units) per correction.")]
    public float maxTeleportUp = 4f;

    [Tooltip("Seconds to pause checks after a successful correction (prevents jitter).")]
    public float cooldownAfterFix = 0.3f;

    [Tooltip("Run the overlap check every N FixedUpdate calls after the first second.")]
    public int checkInterval = 5;

    [Tooltip("Destroy entity after this many lifetime corrections (stuck-loop guard).")]
    public int maxLifetimeFixes = 15;

    // ── Runtime ────────────────────────────────────────────────────────────
    private Collider2D  _col;
    private Rigidbody2D _rb;
    private float _spawnTime;
    private float _cooldownUntil;
    private int   _frameCounter;
    private int   _totalFixes;

    private const int MaxLoggedFixes = 5;

    private void Awake()
    {
        _col       = GetComponent<Collider2D>();
        _rb        = GetComponent<Rigidbody2D>();
        _spawnTime = Time.time;
    }

    private void FixedUpdate()
    {
        if (_col == null || groundLayer.value == 0) return;

        // Grace period
        if (Time.time - _spawnTime < gracePeriod) return;

        // Cooldown after a recent fix
        if (Time.time < _cooldownUntil) return;

        // Throttle: every frame for first 1s, then every N frames
        bool earlyLife = (Time.time - _spawnTime) < 1f;
        if (!earlyLife)
        {
            _frameCounter++;
            if (_frameCounter < checkInterval) return;
            _frameCounter = 0;
        }

        if (!IsStuck()) return;

        // ── Iterative step-up until clear ──────────────────────────────
        Vector3 pos   = transform.position;
        float totalUp = 0f;
        bool  freed   = false;

        while (totalUp < maxTeleportUp)
        {
            pos.y    += stepSize;
            totalUp  += stepSize;
            if (!IsStuckAt(pos)) { freed = true; break; }
        }

        if (!freed) pos.y = transform.position.y + maxTeleportUp; // safety cap

        // Write to both transform and Rigidbody2D (same as LevelInstantiator pattern)
        transform.position = pos;
        if (_rb != null)
        {
            _rb.position = new Vector2(pos.x, pos.y);
            // Zero Y velocity so the entity doesn't immediately fall back in
            _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, 0f);
        }

        _cooldownUntil = Time.time + cooldownAfterFix;
        _totalFixes++;

        if (_totalFixes <= MaxLoggedFixes)
            Debug.Log($"[EntityUnstuck] {name} unstuck → moved up {totalUp:F2}u to {pos} (fix #{_totalFixes})");

        if (_totalFixes >= maxLifetimeFixes)
        {
            Debug.LogWarning($"[EntityUnstuck] {name} exceeded {maxLifetimeFixes} fixes — destroying.");
            Destroy(gameObject);
        }
    }

    // ── Detection helpers ──────────────────────────────────────────────────

    /// <summary>
    /// Returns true if the entity appears embedded in ground at its current position.
    /// Uses a ray-from-above primary check and an overlap secondary check.
    /// </summary>
    private bool IsStuck() => IsStuckAt(transform.position);

    /// <summary>
    /// Tests whether the entity would be stuck if placed at <paramref name="worldPos"/>.
    /// </summary>
    private bool IsStuckAt(Vector3 worldPos)
    {
        // ── Primary: ray from above to find real ground surface ──────────
        // Starts outside the composite shell and hits its outer top face.
        Bounds b = _col.bounds;
        float halfH = b.extents.y;
        float feetY = worldPos.y - halfH;

        Vector2 rayOrigin = new Vector2(worldPos.x, worldPos.y + probeHeight);
        RaycastHit2D hit = Physics2D.Raycast(
            rayOrigin, Vector2.down, probeHeight + 1f, groundLayer);

        if (hit.collider != null)
        {
            float surfaceY = hit.point.y;
            // Entity feet are below the ground surface → embedded
            if (feetY < surfaceY - 0.02f)
                return true;
        }

        // ── Secondary: overlap check for partial-embedding at shell edge ─
        // Shrink by 0.8× so normal surface contact doesn't trigger this.
        Vector2 center = worldPos + (Vector3)_col.offset;
        Vector2 size   = (Vector2)b.size * 0.8f;

        if (_col is CircleCollider2D circ)
            return Physics2D.OverlapCircle(center, circ.radius * 0.8f, groundLayer) != null;
        else
            return Physics2D.OverlapBox(center, size, 0f, groundLayer) != null;
    }
}
