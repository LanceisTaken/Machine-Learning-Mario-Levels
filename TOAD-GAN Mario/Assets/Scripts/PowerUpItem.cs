using UnityEngine;

/// <summary>
/// Abstract base for sliding collectible power-up items (Mushroom, 1-UP, Wind Dash).
///
/// This script is fully self-contained. The prefab only STRICTLY needs:
///   1. SpriteRenderer (for visuals)
///   2. This script (SuperMushroom / OneUpMushroom / WindDashCollectible)
///
/// Everything else (Rigidbody2D, BoxCollider2D) is auto-created in Awake().
///
/// Pickup detection uses Physics2D.OverlapCircle polling in Update() rather
/// than relying on OnTriggerEnter2D or OnCollisionEnter2D, which avoids any
/// issues with physics layers, composite colliders, or missing triggers.
/// </summary>
public abstract class PowerUpItem : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Movement")]
    [Tooltip("Horizontal slide speed in world units/second.")]
    public float moveSpeed = 3f;

    [Tooltip("Y position below which the item is destroyed (fell into a pit).")]
    public float pitDeathY = -10f;

    [Header("Pickup Detection")]
    [Tooltip("Radius to detect the player (world units). Should be >= half the tile size.")]
    public float pickupRadius = 0.5f;

    // ── Runtime ────────────────────────────────────────────────────────────
    protected Rigidbody2D _rb;
    private int _moveDir = 1; // +1 = right, -1 = left
    private bool _collected;

    // ── Lifecycle ──────────────────────────────────────────────────────────
    protected virtual void Awake()
    {
        // ── Auto-create Rigidbody2D ────────────────────────────────────────
        _rb = GetComponent<Rigidbody2D>();
        if (_rb == null)
            _rb = gameObject.AddComponent<Rigidbody2D>();

        _rb.bodyType       = RigidbodyType2D.Dynamic;
        _rb.gravityScale   = 1f;
        _rb.freezeRotation = true;
        _rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;

        // ── Auto-create BoxCollider2D ──────────────────────────────────────
        BoxCollider2D col = GetComponent<BoxCollider2D>();
        if (col == null)
            col = gameObject.AddComponent<BoxCollider2D>();

        col.isTrigger      = false;
        col.usedByComposite = false;

        // Size the collider to match the sprite, or use a sensible default
        SpriteRenderer sr = GetComponent<SpriteRenderer>();
        if (sr != null && sr.sprite != null)
        {
            col.size   = sr.sprite.bounds.size;
            col.offset = Vector2.zero;
        }
        else
        {
            col.size   = new Vector2(0.9f, 0.9f);
            col.offset = Vector2.zero;
        }

        // Frictionless material so it slides smoothly on ground
        if (col.sharedMaterial == null)
        {
            PhysicsMaterial2D mat = new PhysicsMaterial2D("PowerUpSlide")
            {
                friction   = 0f,
                bounciness = 0f
            };
            col.sharedMaterial = mat;
        }

        // Give a small initial upward pop so it visibly emerges from the block
        _rb.linearVelocity = new Vector2(_moveDir * moveSpeed, 3f);

        Debug.Log($"[PowerUpItem] {name} spawned at {transform.position}. " +
                  $"RB={_rb.bodyType}, Gravity={_rb.gravityScale}, ColSize={col.size}");
    }

    protected virtual void Update()
    {
        if (_collected) return;

        // ── Polling-based player pickup ────────────────────────────────────
        // This works regardless of collision layers, trigger setup, or composite issues.
        Collider2D[] hits = Physics2D.OverlapCircleAll(transform.position, pickupRadius);
        foreach (Collider2D hit in hits)
        {
            if (!hit.CompareTag("Player")) continue;

            PlayerController player = hit.GetComponent<PlayerController>()
                                   ?? hit.GetComponentInParent<PlayerController>();
            if (player == null) continue;

            _collected = true;
            Debug.Log($"[PowerUpItem] {name} collected by Player!");
            Collect(player);
            Destroy(gameObject);
            return;
        }

        // ── Pit death ──────────────────────────────────────────────────────
        if (transform.position.y < pitDeathY)
        {
            Debug.Log($"[PowerUpItem] {name} fell into pit at Y={transform.position.y}.");
            Destroy(gameObject);
        }
    }

    protected virtual void FixedUpdate()
    {
        if (_collected) return;

        // Constant horizontal slide; preserve vertical (gravity)
        _rb.linearVelocity = new Vector2(_moveDir * moveSpeed, _rb.linearVelocity.y);
    }

    private void OnCollisionEnter2D(Collision2D col)
    {
        // Don't bounce off the player — pickup is handled in Update()
        if (col.gameObject.CompareTag("Player")) return;

        // Wall bounce: reverse direction on mostly-horizontal contact
        ContactPoint2D contact = col.GetContact(0);
        if (Mathf.Abs(contact.normal.x) > 0.5f)
        {
            _moveDir *= -1;
            Debug.Log($"[PowerUpItem] {name} bounced off wall, now moving {(_moveDir > 0 ? "right" : "left")}.");
        }
    }

    // ── Debug Gizmo ────────────────────────────────────────────────────────
    private void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f, 1f, 0f, 0.4f);
        Gizmos.DrawWireSphere(transform.position, pickupRadius);
    }

    // ── Abstract ───────────────────────────────────────────────────────────

    /// <summary>
    /// Called when the Player touches this item. Apply the power-up effect.
    /// The item is destroyed immediately after this returns.
    /// </summary>
    protected abstract void Collect(PlayerController player);
}
