using UnityEngine;

/// <summary>
/// Basic enemy that patrols left and right, reversing direction on wall contact.
/// Stomped by the player from above (player gets a bounce), killed by star/dash via Die().
/// Uses a two-tier raycast for edge detection:
///   - stairStepDepth: short ray that passses stair steps (ground 1-2 tiles lower).
///   - pitDepth: long ray that catches true pits. Only reverse when BOTH miss.
/// </summary>
public class EnemyPatrol : MonoBehaviour
{
    public float speed = 2f;

    [Tooltip("Layers the enemy treats as walls/ground (set to your Ground layer).")]
    public LayerMask wallLayer;

    [Tooltip("How far ahead to check for walls.")]
    public float wallCheckDistance = 0.4f;

    [Tooltip("How far down (units) to probe for a stair step. ~1.5 tiles lets the enemy walk down one step.")]
    public float stairStepDepth = 1.5f;

    [Tooltip("How far down (units) to probe for a true pit. Only reverse if ground is missing at this range too.")]
    public float pitDepth = 3.5f;

    private Rigidbody2D _rb;
    private Collider2D  _col;
    private int   _direction = -1; // Start moving left
    private float _graceTimer;     // Startup grace period

    /// <summary>Short delay before wall-check raycasting kicks in,
    /// giving the enemy time to pick an open direction on spawn.</summary>
    private const float GracePeriod = 0.3f;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();

        // Never let the rigidbody sleep — prevents enemies from freezing
        if (_rb != null)
            _rb.sleepMode = RigidbodySleepMode2D.NeverSleep;

        // ── Swap BoxCollider2D → CircleCollider2D ──────────────────────
        // A box collider catches its corners on tile edges when walking
        // through 1-tile-wide gaps. A circle slides past smoothly.
        BoxCollider2D box = GetComponent<BoxCollider2D>();
        if (box != null)
        {
            // Size the circle to fit inside the box
            float radius = Mathf.Min(box.size.x, box.size.y) * 0.5f
                         * Mathf.Max(transform.localScale.x, transform.localScale.y);
            Vector2 offset = box.offset;
            DestroyImmediate(box); // must be immediate so _col picks up the new circle, not the dying box

            CircleCollider2D circle = gameObject.AddComponent<CircleCollider2D>();
            circle.radius = radius * 0.9f; // slightly smaller to avoid snagging
            circle.offset = offset;
        }

        _col = GetComponent<Collider2D>();
        _graceTimer = GracePeriod;

        EntityUnstuck unstuck = GetComponent<EntityUnstuck>();
        if (unstuck == null)
            unstuck = gameObject.AddComponent<EntityUnstuck>();
        unstuck.groundLayer = wallLayer;
    }

    private void Start()
    {
        // On spawn, pick the direction that has the most open space.
        // This prevents enemies from freezing when placed right next to a wall.
        PickBestStartDirection();
    }

    private void FixedUpdate()
    {
        if (_rb == null) return;

        // ── Grace period – skip wall checks for a moment after spawning ──
        if (_graceTimer > 0f)
        {
            _graceTimer -= Time.fixedDeltaTime;
            _rb.linearVelocity = new Vector2(_direction * speed, _rb.linearVelocity.y);
            return;
        }

        // ── Wall detection via raycast ──────────────────────────────────
        Vector2 origin   = (Vector2)transform.position;
        float   castDist = wallCheckDistance;

        if (_col != null)
            castDist += _col.bounds.extents.x;

        bool shouldReverse = false;

        RaycastHit2D wallHit = Physics2D.Raycast(
            origin, Vector2.right * _direction, castDist, wallLayer);

        // Reverse direction if we hit a wall, BUT ignore hits on other enemies!
        if (wallHit.collider != null && wallHit.collider.GetComponent<EnemyPatrol>() == null)
            shouldReverse = true;

        // ── Edge detection (two-tier) ───────────────────────────────────
        // Short probe  (stairStepDepth): finds ground on stair step-downs → don't reverse.
        // Long probe   (pitDepth):       finds ground on larger drops → don't reverse.
        // Only reverse when BOTH probes miss → true pit with no ground below.
        if (!shouldReverse)
        {
            float edgeCheckX = _col != null ? _col.bounds.extents.x : 0.4f;
            Vector2 edgeOrigin = origin + Vector2.right * (_direction * edgeCheckX);

            // Offset downward slightly so probes start at the enemy's feet, not centre
            float feetOffsetY = _col != null ? -_col.bounds.extents.y : -0.3f;
            edgeOrigin.y += feetOffsetY;

            RaycastHit2D stairHit = Physics2D.Raycast(edgeOrigin, Vector2.down, stairStepDepth, wallLayer);
            RaycastHit2D pitHit   = Physics2D.Raycast(edgeOrigin, Vector2.down, pitDepth,       wallLayer);

            bool groundNearby = stairHit.collider != null || pitHit.collider != null;
            if (!groundNearby)
                shouldReverse = true;
        }

        if (shouldReverse)
            _direction *= -1;

        // ── Apply movement ─────────────────────────────────────────────
        _rb.linearVelocity = new Vector2(_direction * speed, _rb.linearVelocity.y);
    }

    /// <summary>
    /// Cast rays both left and right to find which direction has more room.
    /// If one side is blocked, pick the other. If both open or both blocked,
    /// default to left.
    /// </summary>
    private void PickBestStartDirection()
    {
        Vector2 origin   = (Vector2)transform.position;
        float   castDist = wallCheckDistance;
        if (_col != null)
            castDist += _col.bounds.extents.x;

        RaycastHit2D hitLeft  = Physics2D.Raycast(origin, Vector2.left,  castDist, wallLayer);
        RaycastHit2D hitRight = Physics2D.Raycast(origin, Vector2.right, castDist, wallLayer);

        bool blockedLeft  = hitLeft.collider  != null && hitLeft.collider.GetComponent<EnemyPatrol>() == null;
        bool blockedRight = hitRight.collider != null && hitRight.collider.GetComponent<EnemyPatrol>() == null;

        if (blockedLeft && !blockedRight)
            _direction = 1;   // go right
        else if (!blockedLeft && blockedRight)
            _direction = -1;  // go left
        // else: both open or both blocked → keep default left
    }

    private void OnCollisionEnter2D(Collision2D col)
    {
        // If we hit another enemy, ignore the collision purely in physics so they walk through each other
        if (col.gameObject.GetComponent<EnemyPatrol>() != null)
        {
            if (_col != null && col.collider != null)
                Physics2D.IgnoreCollision(_col, col.collider);
            return;
        }

        if (!col.gameObject.CompareTag("Player")) return;

        // Player landed on top → stomp kill
        if (col.GetContact(0).normal.y < -0.5f)
        {
            // Award chain kill score through GameManager
            if (GameManager.Instance != null)
            {
                int pts = GameManager.Instance.NextChainKill();
                string label = pts > 0 ? $"+{pts}" : "1-UP!";
                Color  clr   = pts > 0 ? Color.yellow : new Color(0.4f, 1f, 0.4f);
                UIPopup.Show(label, transform.position + Vector3.up * 0.5f, clr);
            }

            // Give player a small bounce
            var playerRb = col.gameObject.GetComponent<Rigidbody2D>();
            if (playerRb != null)
                playerRb.linearVelocity = new Vector2(playerRb.linearVelocity.x, 8f);

            Die();
        }
        else
        {
            // Enemy hit player from the side → damage player
            PlayerController pc = col.gameObject.GetComponent<PlayerController>();
            if (pc != null)
                pc.TakeHit(); // handles HP, invincibility frames, and death
        }
    }

    /// <summary>
    /// Kill this enemy. Called by stomp, star invincibility, and dash.
    /// Score is awarded by the caller so chain counts correctly.
    /// </summary>
    public void Die()
    {
        Destroy(gameObject);
    }
}