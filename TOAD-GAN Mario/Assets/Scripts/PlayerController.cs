using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 6f;
    public float jumpForce = 12f;
    public float fallMultiplier = 2.5f;
    public float lowJumpMultiplier = 2f;

    [Header("Ground Check")]
    public Transform groundCheck;
    public float groundCheckRadius = 0.15f;
    public LayerMask groundLayer;

    private Rigidbody2D _rb;
    private SpriteRenderer _sr;
    private bool _isGrounded;
    private Vector2 _moveInput;
    private bool _jumpPressed;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        _sr = GetComponent<SpriteRenderer>();

        // Fix wall sticking: Create a frictionless physics material if one isn't assigned
        Collider2D col = GetComponent<Collider2D>();
        if (col != null && col.sharedMaterial == null)
        {
            PhysicsMaterial2D noFrictionMat = new PhysicsMaterial2D("NoFriction");
            noFrictionMat.friction = 0f;
            noFrictionMat.bounciness = 0f;
            col.sharedMaterial = noFrictionMat;
        }

        // Safety: ensure the root transform is never negatively scaled.
        // A flipped localScale.x = -1 inverts physics velocity directions,
        // causing the "can only move left" bug. Use SpriteRenderer.flipX instead.
        Vector3 s = transform.localScale;
        if (s.x < 0)
        {
            s.x = Mathf.Abs(s.x);
            transform.localScale = s;
            Debug.LogWarning("[PlayerController] localScale.x was negative — corrected to positive. Use SpriteRenderer.flipX to face directions.");
        }
    }

    private void Update()
    {
        if (Keyboard.current == null) return;

        // ── Input ──────────────────────────────────────────────────────────
        float horizontal =
            (Keyboard.current.dKey.isPressed || Keyboard.current.rightArrowKey.isPressed ? 1f : 0f) +
            (Keyboard.current.aKey.isPressed || Keyboard.current.leftArrowKey.isPressed ? -1f : 0f);

        _moveInput = new Vector2(horizontal, 0f);

        _jumpPressed = Keyboard.current.spaceKey.wasPressedThisFrame ||
                       Keyboard.current.wKey.wasPressedThisFrame ||
                       Keyboard.current.upArrowKey.wasPressedThisFrame;

        // ── Ground detection ───────────────────────────────────────────────
        if (groundCheck != null)
        {
            _isGrounded = Physics2D.OverlapCircle(
                groundCheck.position, groundCheckRadius, groundLayer);
        }
        else
        {
            // Fallback: use a raycast downward from the player's pivot
            _isGrounded = Physics2D.Raycast(
                transform.position, Vector2.down, 0.55f, groundLayer);

            // Only warn once to avoid log spam
#if UNITY_EDITOR
            if (Time.frameCount % 300 == 0)
                Debug.LogWarning("[PlayerController] groundCheck Transform is not assigned! Using raycast fallback. Please assign it in the Inspector.");
#endif
        }

        // ── Jump ───────────────────────────────────────────────────────────
        if (_jumpPressed && _isGrounded)
        {
            _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, jumpForce);
        }

        // ── Sprite flip ────────────────────────────────────────────────────
        if (_sr != null && horizontal != 0f)
            _sr.flipX = horizontal < 0f;
    }

    private void FixedUpdate()
    {
        _rb.linearVelocity = new Vector2(_moveInput.x * moveSpeed, _rb.linearVelocity.y);

        // ── Better Jumping Physics (Mario style) ───────────────────────────
        // Fall faster when moving downward
        if (_rb.linearVelocity.y < 0)
        {
            _rb.linearVelocity += Vector2.up * Physics2D.gravity.y * (fallMultiplier - 1) * Time.fixedDeltaTime;
        }
        // Fall faster if jump button is released early (variable jump height)
        else if (_rb.linearVelocity.y > 0 && !(Keyboard.current.spaceKey.isPressed || Keyboard.current.wKey.isPressed || Keyboard.current.upArrowKey.isPressed))
        {
            _rb.linearVelocity += Vector2.up * Physics2D.gravity.y * (lowJumpMultiplier - 1) * Time.fixedDeltaTime;
        }
    }

    // Draw the ground-check circle in the Scene view for easy debugging
    private void OnDrawGizmosSelected()
    {
        if (groundCheck != null)
        {
            Gizmos.color = _isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(groundCheck.position, groundCheckRadius);
        }
    }
}