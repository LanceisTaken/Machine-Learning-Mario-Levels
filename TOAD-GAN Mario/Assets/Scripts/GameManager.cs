using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Global singleton that tracks score, lives, and chain-kill scoring.
/// Place on a persistent GameObject (DontDestroyOnLoad).
///
/// Score (and optionally lives) labels are auto-created on the Canvas only when
/// <see cref="scoreText"/> is left empty. Assign your own score TextMeshProUGUI
/// and leave <see cref="livesText"/> empty if you only want a score display.
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    // ── State ──────────────────────────────────────────────────────────────
    public int Score { get; private set; }
    public int Lives { get; private set; } = 3;
    public int ChainKillCount { get; private set; }

    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Starting Values")]
    [Tooltip("How many lives the player starts with (used by ResetState).")]
    public int startingLives = 3;

    [Header("UI (optional — auto-created if null)")]
    [Tooltip("TextMeshProUGUI for score. Leave empty to auto-create score + lives at top-centre.")]
    public TMP_Text scoreText;

    [Tooltip("Optional. TextMeshProUGUI for lives; leave empty if you only show score.")]
    public TMP_Text livesText;

    // Chain kill score table: index = kill number (capped at last value)
    private static readonly int[] ChainScores = { 100, 200, 400, 800, 1000, 2000, 4000, 8000 };

    // ── Unity lifecycle ────────────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        Lives = startingLives;
        DontDestroyOnLoad(gameObject);
    }

    private void Start()
    {
        EnsureScoreUI();
        RefreshUI();
    }

    private void OnEnable()
    {
        UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private void OnDisable()
    {
        UnityEngine.SceneManagement.SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene,
                               UnityEngine.SceneManagement.LoadSceneMode mode)
    {
        // Destroyed UI refs compare as null in Unity — recreate missing auto UI only.
        EnsureScoreUI();
        RefreshUI();
    }

    // ── Auto UI creation ───────────────────────────────────────────────────

    /// <summary>
    /// Creates score / lives labels only when needed.
    /// If you assign <see cref="scoreText"/> yourself, nothing is auto-built
    /// (including lives — leave <see cref="livesText"/> empty for score-only HUD).
    /// </summary>
    private void EnsureScoreUI()
    {
        // Fully wired in the Inspector — nothing to spawn at runtime.
        if (scoreText != null && livesText != null)
            return;

        // Custom score placement: do not auto-create an extra lives label or ScoreRow.
        if (scoreText != null)
            return;

        Canvas canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null)
        {
            Debug.LogWarning("[GameManager] No Canvas found — score UI not created.");
            return;
        }

        // Shared container: a horizontal row anchored to the top-centre
        GameObject row = new GameObject("ScoreRow");
        row.transform.SetParent(canvas.transform, false);

        RectTransform rowRect = row.AddComponent<RectTransform>();
        rowRect.anchorMin        = new Vector2(0.5f, 1f);
        rowRect.anchorMax        = new Vector2(0.5f, 1f);
        rowRect.pivot            = new Vector2(0.5f, 1f);
        rowRect.anchoredPosition = new Vector2(0f, -12f);
        rowRect.sizeDelta        = new Vector2(340f, 40f);

        HorizontalLayoutGroup hlg = row.AddComponent<HorizontalLayoutGroup>();
        hlg.spacing              = 30f;
        hlg.childAlignment       = TextAnchor.MiddleCenter;
        hlg.childForceExpandWidth  = false;
        hlg.childForceExpandHeight = false;

        if (scoreText == null)
            scoreText = CreateLabel(row.transform, "ScoreText", 200f);

        if (livesText == null)
            livesText = CreateLabel(row.transform, "LivesText", 110f);

        Debug.Log("[GameManager] Score UI created automatically.");
    }

    private static TMP_Text CreateLabel(Transform parent, string name, float width)
    {
        GameObject go = new GameObject(name);
        go.transform.SetParent(parent, false);

        RectTransform r = go.AddComponent<RectTransform>();
        r.sizeDelta = new Vector2(width, 36f);

        // Semi-transparent dark background for readability over any level
        Image bg = go.AddComponent<Image>();
        bg.color = new Color(0f, 0f, 0f, 0.45f);

        // Child text object (Image and TMP_Text can't share a GameObject)
        GameObject textGo = new GameObject("Text");
        textGo.transform.SetParent(go.transform, false);

        RectTransform tr = textGo.AddComponent<RectTransform>();
        tr.anchorMin  = Vector2.zero;
        tr.anchorMax  = Vector2.one;
        tr.offsetMin  = new Vector2(8f, 0f);
        tr.offsetMax  = new Vector2(-8f, 0f);

        TextMeshProUGUI tmp = textGo.AddComponent<TextMeshProUGUI>();
        tmp.fontSize  = 18f;
        tmp.fontStyle = FontStyles.Bold;
        tmp.color     = Color.white;
        tmp.alignment = TextAlignmentOptions.MidlineLeft;

        return tmp;
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Add points and refresh UI.</summary>
    public void AddScore(int points)
    {
        Score += points;
        RefreshUI();
        Debug.Log($"[GameManager] Score +{points} → {Score}");
    }

    /// <summary>Grant one extra life.</summary>
    public void AddLife()
    {
        Lives++;
        RefreshUI();
        Debug.Log($"[GameManager] 1-UP! Lives → {Lives}");
    }

    /// <summary>
    /// Called when the player loses a stock (HP reached 0). Decrements lives.
    /// </summary>
    /// <returns>True if at least one life remains — respawn the player.</returns>
    public bool LoseLifeOnDeath()
    {
        Lives--;
        Lives = Mathf.Max(0, Lives);
        RefreshUI();
        Debug.Log($"[GameManager] Lost a life. Lives remaining → {Lives}");
        return Lives > 0;
    }

    /// <summary>
    /// Award chain kill score and advance the chain counter.
    /// Returns the points awarded this kill, or 0 when a 1-UP is granted instead.
    /// </summary>
    public int NextChainKill()
    {
        if (ChainKillCount >= ChainScores.Length)
        {
            // Past the score table → award 1-UP
            ChainKillCount++;
            AddLife();
            return 0; // 0 signals "1-UP" to callers
        }

        int pts = ChainScores[ChainKillCount];
        ChainKillCount++;
        AddScore(pts);
        return pts;
    }

    /// <summary>Reset chain kill counter (call when Mario lands or star ends).</summary>
    public void ResetChain()
    {
        ChainKillCount = 0;
    }

    /// <summary>
    /// Resets score, lives, and chain back to starting values.
    /// Call this before reloading the scene so the persistent singleton
    /// starts fresh.
    /// </summary>
    public void ResetState()
    {
        Score          = 0;
        Lives          = startingLives;
        ChainKillCount = 0;
        RefreshUI();
        Debug.Log("[GameManager] State reset.");
    }

    // ── Private helpers ────────────────────────────────────────────────────
    private void RefreshUI()
    {
        if (scoreText != null) scoreText.text = $"Score: {Score}";
        if (livesText  != null) livesText.text  = $"× {Lives}";
    }
}
