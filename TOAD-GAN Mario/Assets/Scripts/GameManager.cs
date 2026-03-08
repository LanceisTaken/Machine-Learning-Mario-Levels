using UnityEngine;
using TMPro;

/// <summary>
/// Global singleton that tracks score, lives, and chain-kill scoring.
/// Place on a persistent GameObject (DontDestroyOnLoad).
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    // ── State ──────────────────────────────────────────────────────────────
    public int Score { get; private set; }
    public int Lives { get; private set; } = 3;
    public int ChainKillCount { get; private set; }

    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("UI (optional)")]
    [Tooltip("Assign a TextMeshProUGUI for the score display.")]
    public TMP_Text scoreText;

    [Tooltip("Assign a TextMeshProUGUI for the lives display.")]
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
        DontDestroyOnLoad(gameObject);
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

    // ── Private helpers ────────────────────────────────────────────────────
    private void RefreshUI()
    {
        if (scoreText != null) scoreText.text = $"SCORE  {Score:D6}";
        if (livesText  != null) livesText.text  = $"× {Lives}";
    }
}
