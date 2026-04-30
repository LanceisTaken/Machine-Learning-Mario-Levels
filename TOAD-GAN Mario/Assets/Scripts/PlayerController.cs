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
    [Tooltip("Hit points at level start.")]
    public int startingHitPoints = 3;
    [Tooltip("Super Mushroom adds +1 HP and cannot push total above this.")]
    public int maxHitPoints = 99;

    [Header("Void / pit")]
    [Tooltip("World Y below which Mario loses a life (void). Tune if your level floor is lower.")]
    public float pitDeathY = -40f;

    // ── UI ─────────────────────────────────────────────────────────────────
    [Header("UI")]
    public UIManager uiManager;

    // ── Dash ───────────────────────────────────────────────────────────────
    [Header("Wind Dash")]
    public float     dashSpeed      = 18f;
    public float     dashDuration   = 0.25f;
    public float     dashCooldown   = 0.8f;
    [Tooltip("Radius used to detect enemies during a dash (world units).")]
    public float     dashKillRadius = 0.5f;
    [Tooltip("Layer mask for enemy detection during dash.")]
    public LayerMask enemyLayer;

    [Header("Double Jump")]
    public AudioClip doubleJumpClip;

    // ── Audio ──────────────────────────────────────────────────────────────
    [Header("Audio (optional)")]
    public AudioSource audioSource;
    public AudioClip   whooshClip;
    public AudioClip   starMusicClip;
    [Tooltip("Normal music AudioSource to pause during star.")]
    public AudioSource normalMusicSource;

    // ── Public State ───────────────────────────────────────────────────────
    /// <summary>True after Mario has collected a Super Mushroom (used by ? blocks for power-up routing).</summary>
    public bool IsBigMario => _hasSuperMushroom;
    public bool IsInvincible { get; private set; }
    public int  DashCharges  { get; private set; }
    public int  DoubleJumpCharges { get; private set; }
    public int  HitPoints    { get; private set; }

    // ── Private ────────────────────────────────────────────────────────────
    private Rigidbody2D    _rb;
    private SpriteRenderer _sr;
    private BoxCollider2D  _col;
    private DashTrail      _dashTrail;

    private static readonly Collider2D[] _dashOverlapBuffer = new Collider2D[8];

    private bool    _isGrounded;
    private bool    _wasGroundedLastFrame;
    private Vector2 _moveInput;
    private bool    _isDashing;
    private bool    _jumpQueued;
    private bool    _hasUsedDoubleJump;
    private float   _dashCooldownTimer;
    private Coroutine _starCoroutine;
    private bool    _hasSuperMushroom;
    private float   _defaultGravityScale;
    private bool    _deathLock;

    // Jump-assist timers
    private float _jumpBufferTimer;   // remembers jump press for a short window
    private float _coyoteTimer;       // allows jumping briefly after leaving ground
    private const float JumpBufferTime = 0.15f;
    private const float CoyoteTime     = 0.1f;

    // ── Lifecycle ──────────────────────────────────────────────────────────
    private void Awake()
    {
        _rb        = GetComponent<Rigidbody2D>();
        _sr        = GetComponent<SpriteRenderer>();
        _col       = GetComponent<BoxCollider2D>();
        _dashTrail = GetComponent<DashTrail>();

        _rb.interpolation = RigidbodyInterpolation2D.Interpolate;
        _defaultGravityScale = _rb.gravityScale;

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
        RefreshAllUi();
    }

    private void Start()
    {
        EnsureUiManager();
        RefreshAllUi();
    }

    private void OnEnable()
    {
        EnsureUiManager();
        RefreshAllUi();
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

        bool jumpInputThisFrame = Keyboard.current.spaceKey.wasPressedThisFrame ||
                                  Keyboard.current.wKey.wasPressedThisFrame      ||
                                  Keyboard.current.upArrowKey.wasPressedThisFrame;

        // ── Ground Detection ───────────────────────────────────────────────
        _wasGroundedLastFrame = _isGrounded;
        _isGrounded = groundCheck != null
            ? Physics2D.OverlapCircle(groundCheck.position, groundCheckRadius, groundLayer)
            : Physics2D.Raycast(transform.position, Vector2.down, 0.55f, groundLayer);

        // Reset stomp/kill chain only on the frame Mario lands (airborne → grounded)
        if (_isGrounded && !_wasGroundedLastFrame && GameManager.Instance != null)
        {
            GameManager.Instance.ResetChain();
            _hasUsedDoubleJump = false;
        }

#if UNITY_EDITOR
        if (groundCheck == null && Time.frameCount % 300 == 0)
            Debug.LogWarning("[PlayerController] groundCheck not assigned — using raycast fallback.");
#endif

        // ── Coyote time: allow jumping briefly after walking off a ledge ──
        if (_isGrounded)
            _coyoteTimer = CoyoteTime;
        else
            _coyoteTimer -= Time.deltaTime;

        // ── Jump buffering: remember press for a short window ─────────────
        if (jumpInputThisFrame)
            _jumpBufferTimer = JumpBufferTime;
        else
            _jumpBufferTimer -= Time.deltaTime;

        // ── Jump ───────────────────────────────────────────────────────────
        bool canJump = _coyoteTimer > 0f;
        if (_jumpBufferTimer > 0f && canJump)
        {
            _jumpQueued      = true;  // executed in FixedUpdate to stay in sync with physics
            _jumpBufferTimer = 0f;
            _coyoteTimer     = 0f;
        }
        else if (jumpInputThisFrame && !_isGrounded && !_hasUsedDoubleJump && DoubleJumpCharges > 0)
        {
            _jumpQueued        = true;
            _jumpBufferTimer   = 0f;
            _hasUsedDoubleJump = true;
            DoubleJumpCharges--;
            SyncUiJumps();

            if (audioSource != null && doubleJumpClip != null)
                audioSource.PlayOneShot(doubleJumpClip);

            if (DoubleJumpCharges <= 0)
            {
                UIPopup.Show("JUMP GONE", transform.position + Vector3.up, new Color(1f, 0.5f, 0.5f));
                Debug.Log("[PlayerController] Double jump power-up expired.");
            }
        }

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
        if (isActiveAndEnabled && transform.position.y < pitDeathY)
        {
            DieFromVoid();
            return;
        }

        if (_isDashing) return;

        if (_jumpQueued)
        {
            _rb.linearVelocity = new Vector2(_rb.linearVelocity.x, jumpForce);
            _jumpQueued = false;
        }

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
    /// Called by the Super Mushroom. Marks Mario as big for ? blocks and adds +1 HP up to <see cref="maxHitPoints"/>.
    /// </summary>
    public void GrowBig()
    {
        _hasSuperMushroom = true;
        int cap = Mathf.Max(maxHitPoints, startingHitPoints);
        if (HitPoints < cap)
            HitPoints = Mathf.Min(HitPoints + 1, cap);
        SyncUiHealth();
        Debug.Log($"[PlayerController] Super Mushroom collected — HP now {HitPoints}.");
    }

    /// <summary>
    /// Called when Mario takes damage. Loses 1 HP and gets brief i-frames.
    /// </summary>
    public void TakeHit()
    {
        if (IsInvincible) return;

        HitPoints--;
        SyncUiHealth();
        Debug.Log($"[PlayerController] Hit! HP now {HitPoints}.");

        if (HitPoints > 0)
            StartCoroutine(HitInvincibilityCoroutine());
        else
            HandleOutOfHealth();
    }

    /// <summary>Lose a stock from falling into the void (bypasses star / hit i-frames).</summary>
    private void DieFromVoid()
    {
        if (_deathLock || !isActiveAndEnabled) return;

        HitPoints = 0;
        SyncUiHealth();
        HandleOutOfHealth();
    }

    private void HandleOutOfHealth()
    {
        if (_deathLock) return;
        _deathLock = true;

        Debug.Log("[PlayerController] Game Over.");
        Time.timeScale = 0f;

        RestartManager.EnsureExists();
        RestartManager.Instance.ShowRestartButton();
    }

    private void RespawnAfterLifeLost()
    {
        StopAllCoroutines();
        _starCoroutine = null;
        _isDashing = false;
        _dashCooldownTimer = 0f;
        _jumpQueued = false;
        _jumpBufferTimer = 0f;
        _coyoteTimer = 0f;
        _hasUsedDoubleJump = false;

        _dashTrail?.StopTrail();

        if (_rb != null)
        {
            _rb.gravityScale = _defaultGravityScale;
            _rb.linearVelocity = Vector2.zero;
            _rb.angularVelocity = 0f;
        }

        if (audioSource != null) audioSource.Stop();
        if (normalMusicSource != null) normalMusicSource.UnPause();

        IsInvincible = false;
        if (_sr != null)
        {
            _sr.enabled = true;
            _sr.color = Color.white;
        }

        HitPoints = Mathf.Max(1, startingHitPoints);
        _hasSuperMushroom = false;

        LevelInstantiator li = FindFirstObjectByType<LevelInstantiator>();
        Vector3 spawn = li != null ? li.LastPlayerSpawnWorld : transform.position;
        if (_rb != null)
            _rb.position = new Vector2(spawn.x, spawn.y);
        transform.position = spawn;

        GameManager.Instance?.ResetChain();

        EnsureUiManager();
        RefreshAllUi();
        RestartManager.Instance?.HideRestartButton();
        Debug.Log("[PlayerController] Respawned after losing a life.");
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
        DashCharges += Mathf.Max(0, charges);
        SyncUiDashes();
        Debug.Log($"[PlayerController] Dash +{charges} (total {DashCharges}).");
    }

    /// <summary>Grant double-jump charges.</summary>
    public void GrantDoubleJump(int charges = 2)
    {
        DoubleJumpCharges += Mathf.Max(0, charges);
        _hasUsedDoubleJump = false;
        SyncUiJumps();
        Debug.Log($"[PlayerController] Double-jump +{charges} (total {DoubleJumpCharges}).");
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
        string label = pts > 0 ? $"+{pts}" : "1-UP!";
        Color  clr   = pts > 0 ? Color.yellow : new Color(0.4f, 1f, 0.4f);
        UIPopup.Show(label, other.transform.position, clr);
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
            if (normalMusicSource != null) normalMusicSource.Pause();
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
        if (normalMusicSource != null) normalMusicSource.UnPause();

        Debug.Log("[PlayerController] Star power ended.");
    }

    private IEnumerator DashCoroutine()
    {
        DashCharges--;
        SyncUiDashes();
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
        int hitCount = Physics2D.OverlapCircleNonAlloc(
            transform.position, dashKillRadius, _dashOverlapBuffer, enemyLayer);

        for (int i = 0; i < hitCount; i++)
        {
            Collider2D hit = _dashOverlapBuffer[i];
            EnemyPatrol enemy = hit.GetComponent<EnemyPatrol>()
                             ?? hit.GetComponentInParent<EnemyPatrol>();
            if (enemy == null) continue;

            int pts = GameManager.Instance != null ? GameManager.Instance.NextChainKill() : 100;
            string label = pts > 0 ? $"+{pts}" : "1-UP!";
            Color  clr   = pts > 0 ? Color.yellow : new Color(0.4f, 1f, 0.4f);
            UIPopup.Show(label, hit.transform.position, clr);
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

    private void EnsureUiManager()
    {
        if (uiManager != null) return;

        UIManager[] all = Object.FindObjectsByType<UIManager>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        foreach (UIManager ui in all)
        {
            if (ui != null && ui.healthText != null)
            {
                uiManager = ui;
                return;
            }
        }

        if (all.Length > 0)
            uiManager = all[0];
    }

    private void RefreshAllUi()
    {
        SyncUiHealth();
        SyncUiDashes();
        SyncUiJumps();
    }

    private void SyncUiHealth()
    {
        if (uiManager != null) uiManager.UpdateHealth(HitPoints);
    }

    private void SyncUiDashes()
    {
        if (uiManager != null) uiManager.UpdateDashes(DashCharges);
    }

    private void SyncUiJumps()
    {
        if (uiManager != null) uiManager.UpdateJumps(DoubleJumpCharges);
    }
}