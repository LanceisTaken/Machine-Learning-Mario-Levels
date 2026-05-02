using System.Collections;
using UnityEngine;

/// <summary>
/// Super Star (Starman) power-up.
///
/// Movement: unlike mushrooms, the star bounces upward in arcs using
/// periodic upward impulses while drifting rightward.
///
/// On collection: grants Mario temporary invincibility for `starDuration`
/// seconds and awards 1000 points.
/// </summary>
[RequireComponent(typeof(Rigidbody2D))]
public class StarMan : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Movement")]
    public float moveSpeed     = 4f;
    public float bounceForce   = 8f;
    [Tooltip("How often (in seconds) the star bounces upward.")]
    public float bounceInterval = 0.45f;
    public float pitDeathY     = -10f;

    [Header("Power-up")]
    public float starDuration  = 10f;

    // ── Runtime ────────────────────────────────────────────────────────────
    private Rigidbody2D _rb;
    private float _bounceTimer;

    // ── Unity lifecycle ────────────────────────────────────────────────────
    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        _rb.gravityScale   = 1f;
        _rb.freezeRotation = true;

        // Give an initial upward pop when spawned from a block
        _rb.linearVelocity = new Vector2(moveSpeed, bounceForce);
    }

    private void FixedUpdate()
    {
        // Keep constant rightward movement
        _rb.linearVelocity = new Vector2(moveSpeed, _rb.linearVelocity.y);

        // Periodic bounce impulse
        _bounceTimer += Time.fixedDeltaTime;
        if (_bounceTimer >= bounceInterval)
        {
            _bounceTimer = 0f;
            // Only bounce upward when falling or near ground to avoid cancelling jumps
            if (_rb.linearVelocity.y <= 0.5f)
                _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, bounceForce);
        }

        // Pit death
        if (transform.position.y < pitDeathY)
            Destroy(gameObject);
    }

    private void OnCollisionEnter2D(Collision2D col)
    {
        if (col.gameObject.CompareTag("Player")) return;

        // Reverse on wall hit
        ContactPoint2D contact = col.GetContact(0);
        if (Mathf.Abs(contact.normal.x) > 0.5f)
            moveSpeed = -moveSpeed;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (!other.CompareTag("Player")) return;

        PlayerController player = other.GetComponent<PlayerController>()
                               ?? other.GetComponentInParent<PlayerController>();
        if (player == null) return;

        player.ActivateStar(starDuration);

        if (GameManager.Instance != null)
            GameManager.Instance.AddScore(1000);

        UIPopup.Show("+1000", transform.position, new Color(1f, 0.85f, 0f));
        Destroy(gameObject);
    }
}
