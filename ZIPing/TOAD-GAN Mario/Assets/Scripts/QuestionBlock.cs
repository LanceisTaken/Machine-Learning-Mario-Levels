using System.Collections;
using UnityEngine;

/// <summary>
/// Attach to the '?' (Question Block) tile prefab.
///
/// When Mario hits the block from below, it spawns the appropriate power-up:
///   • Small Mario  → Super Mushroom
///   • Big Mario    → random Wind Dash or Jump Power-Up (50/50 when both prefabs assigned)
///
/// After being hit, the block is "used" (sprite changes to grey/empty) and
/// cannot be hit again.
///
/// IMPORTANT – CompositeCollider2D compatibility
/// ------------------------------------------------
/// LevelInstantiator adds a CompositeCollider2D to the level root, which
/// would absorb child BoxCollider2D components and suppress OnCollisionEnter2D.
/// This script opts out of the composite in Awake() and instead uses its own
/// Rigidbody2D (Static) so collisions fire correctly.
///
/// Prefab setup
/// ------------
///  • Assign superMushroomPrefab, windDashPrefab, jumpPowerUpPrefab in the Inspector.
///  • Optionally set usedSprite for the depleted appearance.
///  • The block needs a BoxCollider2D + Rigidbody2D (Static) on the prefab itself.
///  • The Player must be tagged "Player".
/// </summary>
public class QuestionBlock : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Power-up Prefabs")]
    [Tooltip("Spawned when small Mario hits the block.")]
    public GameObject superMushroomPrefab;

    [Tooltip("Spawned when big Mario hits the block (random pool with jump power-up).")]
    public GameObject windDashPrefab;

    [Tooltip("Spawned when big Mario hits the block (random pool with wind dash).")]
    public GameObject jumpPowerUpPrefab;

    [Header("Visual")]
    [Tooltip("Sprite to display after the block has been used (grey/empty block). Assign this in the Inspector.")]
    public Sprite usedSprite;

    [Tooltip("Colour tint applied to the used sprite (default grey to look depleted).")]
    public Color usedTint = new Color(0.55f, 0.55f, 0.55f, 1f);

    [Tooltip("How many times the sprite flashes white before settling on usedSprite.")]
    public int flashCount = 3;

    [Tooltip("Duration (seconds) of each flash half-cycle.")]
    public float flashInterval = 0.05f;

    [Tooltip("How far above the block centre the item spawns.")]
    public float spawnOffset = 1.2f;

    [Header("Bounce Animation")]
    [Tooltip("How high the block bounces (world units).")]
    public float bounceHeight   = 0.25f;
    [Tooltip("Duration of the bounce animation in seconds.")]
    public float bounceDuration = 0.12f;

    // ── State ──────────────────────────────────────────────────────────────
    private bool           _used;
    private SpriteRenderer _sr;
    private Vector3        _originPos;

    // ── Unity lifecycle ────────────────────────────────────────────────────
    private void Awake()
    {
        _sr        = GetComponent<SpriteRenderer>();
        _originPos = transform.position;

        // ── Opt out of the parent's CompositeCollider2D ────────────────────
        // LevelInstantiator adds a CompositeCollider2D to the level root.
        // Any child BoxCollider2D with usedByComposite=true gets absorbed and
        // its MonoBehaviour collision callbacks stop firing.
        // Setting usedByComposite=false on our own colliders keeps them
        // independent so OnCollisionEnter2D works normally.
        foreach (BoxCollider2D bc in GetComponents<BoxCollider2D>())
            bc.usedByComposite = false;

        // Ensure we have a Rigidbody2D (Static) on this object so Unity can
        // dispatch collision callbacks. LevelInstantiator's Rigidbody2D is on
        // the root, not on this child.
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody2D>();
            rb.bodyType = RigidbodyType2D.Static;
        }
    }

    private void OnCollisionEnter2D(Collision2D col)
    {
        if (_used) return;
        if (!col.gameObject.CompareTag("Player")) return;

        // ── Detect hit from BELOW ──────────────────────────────────────────
        // In OnCollisionEnter2D on the BLOCK, the contact normal points
        // FROM the player (col.gameObject) TOWARD the block (this).
        // When Mario jumps up and hits the bottom of the block, that vector
        // points UPWARD → normal.y is POSITIVE (~+1).
        // We keep only hits where the normal is pointing strongly upward.
        ContactPoint2D contact = col.GetContact(0);
        if (contact.normal.y < 0.7f) return;   // not a bottom hit → ignore

        Hit(col.gameObject);
    }

    // ── Private helpers ────────────────────────────────────────────────────
    private void Hit(GameObject playerGo)
    {
        _used = true;

        // Award 200 pts for hitting the ? block (same as coin from block in SMB)
        if (GameManager.Instance != null)
            GameManager.Instance.AddScore(200);
        UIPopup.Show("+200", transform.position + Vector3.up * 0.5f, Color.yellow);

        // Swap to used/depleted sprite with a brief flash transition
        StartCoroutine(SpriteTransition());

        // Choose which power-up to spawn
        PlayerController player = playerGo.GetComponent<PlayerController>()
                                 ?? playerGo.GetComponentInParent<PlayerController>();

        GameObject prefabToSpawn;
        if (player == null || !player.IsBigMario)
            prefabToSpawn = superMushroomPrefab;
        else
        {
            bool haveWind = windDashPrefab != null;
            bool haveJump = jumpPowerUpPrefab != null;
            if (haveWind && haveJump)
                prefabToSpawn = Random.value < 0.5f ? windDashPrefab : jumpPowerUpPrefab;
            else if (haveWind)
                prefabToSpawn = windDashPrefab;
            else
                prefabToSpawn = jumpPowerUpPrefab;
        }
        if (prefabToSpawn != null)
        {
            Vector3 spawnPos = transform.position + Vector3.up * spawnOffset;
            Instantiate(prefabToSpawn, spawnPos, Quaternion.identity);
        }
        else
        {
            Debug.LogWarning("[QuestionBlock] No prefab assigned! Check Inspector slots.", this);
        }

        StartCoroutine(BounceAnimation());
    }

    private IEnumerator BounceAnimation()
    {
        float elapsed = 0f;
        while (elapsed < bounceDuration)
        {
            float t = elapsed / bounceDuration;
            transform.position = _originPos + Vector3.up * (Mathf.Sin(t * Mathf.PI) * bounceHeight);
            elapsed += Time.deltaTime;
            yield return null;
        }
        transform.position = _originPos;
    }

    /// <summary>
    /// Flashes the block white several times, then switches to the used sprite with a grey tint.
    /// If no usedSprite is assigned the block simply flashes and stays on its original sprite.
    /// </summary>
    private IEnumerator SpriteTransition()
    {
        if (_sr == null) yield break;

        Color originalColor = _sr.color;

        for (int i = 0; i < flashCount; i++)
        {
            _sr.color = Color.white;
            yield return new WaitForSeconds(flashInterval);
            _sr.color = originalColor;
            yield return new WaitForSeconds(flashInterval);
        }

        if (usedSprite != null)
        {
            _sr.sprite = usedSprite;
            _sr.color  = usedTint;
        }
    }
}
