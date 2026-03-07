using System.Collections;
using UnityEngine;

/// <summary>
/// Attach to the '?' (Question Block) tile prefab.
///
/// When Mario hits the block from below, it spawns the appropriate power-up:
///   • Small Mario  → Super Mushroom
///   • Big Mario    → Wind Dash Collectible
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
///  • Assign superMushroomPrefab, windDashPrefab in the Inspector.
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

    [Tooltip("Spawned when big Mario hits the block.")]
    public GameObject windDashPrefab;

    [Header("Visual")]
    [Tooltip("Sprite to display after the block has been used (grey block).")]
    public Sprite usedSprite;

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

        // Swap to used/depleted sprite
        if (_sr != null && usedSprite != null)
            _sr.sprite = usedSprite;

        // Choose which power-up to spawn
        PlayerController player = playerGo.GetComponent<PlayerController>();
        bool isBig = player != null && player.IsBigMario;

        GameObject prefabToSpawn = isBig ? windDashPrefab : superMushroomPrefab;
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
}
