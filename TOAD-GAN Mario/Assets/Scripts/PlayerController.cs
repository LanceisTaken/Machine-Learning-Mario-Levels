using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    // ── Movement ───────────────────────────────────────────────────────────
    [Header("Movement")]
    public float moveSpeed         = 6f;
    public float jumpForce         = 12f;
    public float fallMultiplier    = 2.5f;
    public float lowJumpMultiplier = 2f;

    // ── Ground Check ───────────────────────────────────────────────────────
    [Header("Ground Check")]
    public Transform groundCheck;
    public float     groundCheckRadius = 0.15f;
    public LayerMask groundLayer;

    // ── Health ─────────────────────────────────────────────────────────────
    [Header("Health")]
    [Tooltip("Starting hit points. The Super Mushroom grants +1 HP (max 2).")]
    public int startingHitPoints = 1;

    // ── Dash ───────────────────────────────────────────────────────────────
    [Header("Wind Dash")]
    public float     dashSpeed      = 18f;
    public float     dashDuration   = 0.25f;
    public float     dashCooldown   = 0.8f;
    [Tooltip("Radius used to detect enemies during a dash (world units).")]
    public float     dashKillRadius = 0.5f;
    [Tooltip("Layer mask for enemy detection during dash.")]
    public LayerMask enemyLayer;

    // ── Audio ──────────────────────────────────────────────────────────────
    [Header("Audio (optional)")]
    public AudioSource audioSource;
    public AudioClip   whooshClip;
    public AudioClip   starMusicClip;
    [Tooltip("Normal music AudioSource to pause during star.")]
    public AudioSource normalMusicSource;

    // ── Public State ───────────────────────────────────────────────────────
    /// <summary>True when Mario has collected a Super Mushroom (HP > 1).</summary>
    public bool IsBigMario   => HitPoints > 1;
    public bool IsInvincible { get; private set; }
    public int  DashCharges  { get; private set; }
    public int  HitPoints    { get; private set; }

    // ── Private ────────────────────────────────────────────────────────────
    private Rigidbody2D    _rb;
    private SpriteRenderer _sr;
    private BoxCollider2D  _col;
    private DashTrail      _dashTrail;

    private bool    _isGrounded;
    private Vector2 _moveInput;
    private bool    _jumpPressed;
    private bool    _isDashing;
    private float   _dashCooldownTimer;
    private Coroutine _starCoroutine;

    // ── Lifecycle ──────────────────────────────────────────────────────────
    private void Awake()
    {
        _rb        = GetComponent<Rigidbody2D>();
        _sr        = GetComponent<SpriteRenderer>();
        _col       = GetComponent<BoxCollider2D>();
        _dashTrail = GetComponent<DashTrail>();

        // Frictionless material so player doesn't stick to walls
        if (_col != null && _col.sharedMaterial == null)
        {
            PhysicsMaterial2D noFriction = new PhysicsMaterial2D("NoFriction")
            {
                friction   = 0f,
                bounciness = 0f
            };
            _col.sharedMaterial = noFriction;
        }

        // Ensure scale.x is never negative (use SpriteRenderer.flipX for direction)
        Vector3 s = transform.localScale;
        if (s.x < 0) { s.x = Mathf.Abs(s.x); transform.localScale = s; }

        HitPoints = Mathf.Max(1, startingHitPoints);
    }

    private void Update()
    {
        if (Keyboard.current == null) return;
        if (_isDashing) return;

        // ── Directional Input ──────────────────────────────────────────────
        float horizontal =
            (Keyboard.current.dKey.isPressed || Keyboard.current.rightArrowKey.isPressed ?  1f : 0f) +
            (Keyboard.current.aKey.isPressed || Keyboard.current.leftArrowKey.isPressed  ? -1f : 0f);

        _moveInput = new Vector2(horizontal, 0f);

        _jumpPressed = Keyboard.current.spaceKey.wasPressedThisFrame ||
                       Keyboard.current.wKey.wasPressedThisFrame      ||
                       Keyboard.current.upArrowKey.wasPressedThisFrame;

        // ── Ground Detection ───────────────────────────────────────────────
        _isGrounded = groundCheck != null
            ? Physics2D.OverlapCircle(groundCheck.position, groundCheckRadius, groundLayer)
            : Physics2D.Raycast(transform.position, Vector2.down, 0.55f, groundLayer);

#if UNITY_EDITOR
        if (groundCheck == null && Time.frameCount % 300 == 0)
            Debug.LogWarning("[PlayerController] groundCheck not assigned — using raycast fallback.");
#endif

        // ── Jump ───────────────────────────────────────────────────────────
        if (_jumpPressed && _isGrounded)
            _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, jumpForce);

        // ── Sprite Flip ────────────────────────────────────────────────────
        if (_sr != null && horizontal != 0f)
            _sr.flipX = horizontal < 0f;

        // ── Dash Input (Left/Right Shift) ──────────────────────────────────
        _dashCooldownTimer -= Time.deltaTime;
        bool dashPressed = Keyboard.current.leftShiftKey.wasPressedThisFrame ||
                           Keyboard.current.rightShiftKey.wasPressedThisFrame;
        if (dashPressed && DashCharges > 0 && _dashCooldownTimer <= 0f)
            StartCoroutine(DashCoroutine());
    }

    private void FixedUpdate()
    {
        if (_isDashing) return;

        _rb.linearVelocity = new Vector2(_moveInput.x * moveSpeed, _rb.linearVelocity.y);

        if (_rb.linearVelocity.y < 0)
        {
            _rb.linearVelocity += Vector2.up * Physics2D.gravity.y * (fallMultiplier - 1) * Time.fixedDeltaTime;
        }
        else if (_rb.linearVelocity.y > 0 &&
                 !(Keyboard.current.spaceKey.isPressed ||
                   Keyboard.current.wKey.isPressed     ||
                   Keyboard.current.upArrowKey.isPressed))
        {
            _rb.linearVelocity += Vector2.up * Physics2D.gravity.y * (lowJumpMultiplier - 1) * Time.fixedDeltaTime;
        }
    }

    // ── Power-Up API ───────────────────────────────────────────────────────

    /// <summary>
    /// Called by the Super Mushroom.
    /// Gives Mario an extra hit point (up to 2). No visual or collider changes.
    /// </summary>
    public void GrowBig()
    {
        if (HitPoints >= 2) return; // already has the bonus HP
        HitPoints = 2;
        Debug.Log("[PlayerController] Super Mushroom collected — HP is now 2.");
    }

    /// <summary>
    /// Called when Mario is hit while at 2 HP. Drops back to 1 HP with brief
    /// invincibility frames to prevent instant double-hit.
    /// </summary>
    public void TakeHit()
    {
        if (IsInvincible) return;

        HitPoints--;
        Debug.Log($"[PlayerController] Hit! HP now {HitPoints}.");

        if (HitPoints <= 0)
        {
            Debug.Log("[PlayerController] Mario is dead! (Add game-over logic here.)");
            // TODO: trigger death / respawn
        }
        else
        {
            // Brief invincibility so the player can't be hit twice in one frame
            StartCoroutine(HitInvincibilityCoroutine());
        }
    }

    /// <summary>Activate star invincibility for <paramref name="duration"/> seconds.</summary>
    public void ActivateStar(float duration = 10f)
    {
        if (_starCoroutine != null) StopCoroutine(_starCoroutine);
        _starCoroutine = StartCoroutine(StarCoroutine(duration));
    }

    /// <summary>Grant wind-dash charges.</summary>
    public void GrantDash(int charges = 3)
    {
        DashCharges = charges;
        Debug.Log($"[PlayerController] Dash granted: {charges} charges.");
    }

    // ── Enemy contact while invincible ─────────────────────────────────────
    private void OnTriggerEnter2D(Collider2D other)
    {
        if (!IsInvincible) return;
        if (!other.CompareTag("Enemy")) return;

        EnemyPatrol enemy = other.GetComponent<EnemyPatrol>()
                         ?? other.GetComponentInParent<EnemyPatrol>();
        if (enemy == null) return;

        int pts = GameManager.Instance != null ? GameManager.Instance.NextChainKill() : 100;
        UIPopup.Show($"+{pts}", other.transform.position, Color.yellow);
        enemy.Die();
    }

    // ── Coroutines ─────────────────────────────────────────────────────────

    private IEnumerator HitInvincibilityCoroutine()
    {
        // 1.5-second invincibility + flicker after taking a hit
        IsInvincible = true;
        float elapsed = 0f;
        while (elapsed < 1.5f)
        {
            if (_sr != null) _sr.enabled = !_sr.enabled;
            elapsed += 0.1f;
            yield return new WaitForSeconds(0.1f);
        }
        if (_sr != null) _sr.enabled = true;
        IsInvincible = false;
    }

    private IEnumerator StarCoroutine(float duration)
    {
        IsInvincible = true;
        GameManager.Instance?.ResetChain();

        if (audioSource != null && starMusicClip != null)
        {
            normalMusicSource?.Pause();
            audioSource.clip = starMusicClip;
            audioSource.loop = true;
            audioSource.Play();
        }

        float elapsed = 0f;
        Color[] flickerColors = { Color.yellow, Color.red, Color.white, Color.cyan };
        int flickerIdx = 0;

        while (elapsed < duration)
        {
            if (_sr != null) _sr.color = flickerColors[flickerIdx % flickerColors.Length];
            flickerIdx++;
            elapsed += 0.1f;
            yield return new WaitForSeconds(0.1f);
        }

        if (_sr != null) _sr.color = Color.white;
        IsInvincible = false;
        GameManager.Instance?.ResetChain();

        if (audioSource != null) audioSource.Stop();
        normalMusicSource?.UnPause();

        Debug.Log("[PlayerController] Star power ended.");
    }

    private IEnumerator DashCoroutine()
    {
        DashCharges--;
        _isDashing = true;
        _dashCooldownTimer = dashCooldown;

        if (audioSource != null && whooshClip != null)
            audioSource.PlayOneShot(whooshClip);

        _dashTrail?.StartTrail();

        float dir = _sr != null && _sr.flipX ? -1f : 1f;

        float origGravity = _rb.gravityScale;
        _rb.gravityScale   = 0f;
        _rb.linearVelocity = new Vector2(dir * dashSpeed, 0f);

        float elapsed = 0f;
        while (elapsed < dashDuration)
        {
            KillEnemiesInRange();
            elapsed += Time.deltaTime;
            yield return null;
        }

        _rb.gravityScale   = origGravity;
        _rb.linearVelocity = new Vector2(0f, _rb.linearVelocity.y);

        _dashTrail?.StopTrail();
        _isDashing = false;

        if (DashCharges <= 0)
        {
            UIPopup.Show("DASH GONE", transform.position + Vector3.up, new Color(1f, 0.5f, 0.5f));
            Debug.Log("[PlayerController] Dash power-up expired.");
        }
    }

    private void KillEnemiesInRange()
    {
        Collider2D[] hits = Physics2D.OverlapCircleAll(
            transform.position, dashKillRadius, enemyLayer);

        foreach (Collider2D hit in hits)
        {
            EnemyPatrol enemy = hit.GetComponent<EnemyPatrol>()
                             ?? hit.GetComponentInParent<EnemyPatrol>();
            if (enemy == null) continue;

            int pts = GameManager.Instance != null ? GameManager.Instance.NextChainKill() : 100;
            UIPopup.Show($"+{pts}", hit.transform.position, Color.yellow);
            enemy.Die();
        }
    }

    // ── Gizmos ─────────────────────────────────────────────────────────────
    private void OnDrawGizmosSelected()
    {
        if (groundCheck != null)
        {
            Gizmos.color = _isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(groundCheck.position, groundCheckRadius);
        }

        Gizmos.color = new Color(0f, 0.5f, 1f, 0.3f);
        Gizmos.DrawWireSphere(transform.position, dashKillRadius);
    }
}