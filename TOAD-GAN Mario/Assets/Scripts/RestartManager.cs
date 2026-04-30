using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;

/// <summary>
/// Manages the in-game restart button.
///
/// Setup (one-time):
///   Attach this component to any persistent GameObject (e.g. the Canvas or
///   GameManager object in SampleScene).  No manual wiring is required —
///   the button is created automatically at runtime and positioned in the
///   top-right corner of the screen.
///
///   Optionally, assign an existing <see cref="UnityEngine.UI.Button"/> to
///   <see cref="restartButton"/> in the Inspector to skip auto-creation and
///   use a hand-crafted button instead.
///
/// Behaviour:
///   • Clicking the button resets <see cref="GameManager"/> state, then
///     reloads the active scene.
///   • <see cref="ShowRestartButton"/> / <see cref="HideRestartButton"/> let
///     other scripts (e.g. <see cref="PlayerController"/> on death) make the
///     button visible or invisible at the right moment.
///   • The button is hidden by default so it does not obstruct gameplay;
///     pass <c>visibleByDefault = true</c> in the Inspector if you always
///     want it on screen.
/// </summary>
public class RestartManager : MonoBehaviour
{
    // ── Singleton ──────────────────────────────────────────────────────────
    public static RestartManager Instance { get; private set; }

    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Button (optional — auto-created if null)")]
    [Tooltip("Assign an existing Button to skip auto-creation.")]
    public Button restartButton;

    [Tooltip("Show the restart button as soon as the scene loads.")]
    public bool visibleByDefault = false;

    [Header("Auto-created Button Style")]
    [Tooltip("Background colour of the auto-created button.")]
    public Color buttonColor = new Color(0.85f, 0.2f, 0.2f, 0.92f);

    [Tooltip("Label text colour.")]
    public Color labelColor = Color.white;

    [Tooltip("Button size in pixels.")]
    public Vector2 buttonSize = new Vector2(160f, 50f);

    [Tooltip("Distance from the top-right corner in pixels.")]
    public Vector2 cornerOffset = new Vector2(20f, 20f);

    [Header("Death overlay")]
    [Tooltip("Colour of the full-screen tint shown on death.")]
    public Color overlayColor = new Color(0.6f, 0f, 0f, 0.45f);

    private bool _restartClickWired;
    private Image _deathOverlay;

    // ── Lifecycle ──────────────────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    private void Start()
    {
        EnsureRestartUiReady();
        SetButtonVisible(visibleByDefault);
    }

    /// <summary>
    /// Creates the button and wires the click listener if needed.
    /// Safe to call before <see cref="Start"/> (e.g. game over on first frame).
    /// </summary>
    private void EnsureRestartUiReady()
    {
        if (restartButton == null)
            restartButton = CreateRestartButton();

        if (!_restartClickWired && restartButton != null)
        {
            restartButton.onClick.AddListener(OnRestartClicked);
            _restartClickWired = true;
        }
    }

    /// <summary>If no RestartManager exists yet, add one so death UI always works.</summary>
    public static void EnsureExists()
    {
        if (Instance != null) return;
        new GameObject("RestartManager").AddComponent<RestartManager>();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Show the red death overlay and restart button.</summary>
    public void ShowRestartButton()
    {
        EnsureRestartUiReady();
        SetOverlayVisible(true);
        SetButtonVisible(true);
    }

    /// <summary>Hide the overlay and restart button.</summary>
    public void HideRestartButton()
    {
        SetOverlayVisible(false);
        SetButtonVisible(false);
    }

    // ── Private helpers ────────────────────────────────────────────────────

    private void OnRestartClicked()
    {
        SetOverlayVisible(false);
        Time.timeScale = 1f;

        if (GameManager.Instance != null)
            Destroy(GameManager.Instance.gameObject);

        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    private void SetButtonVisible(bool visible)
    {
        if (restartButton != null)
            restartButton.gameObject.SetActive(visible);
    }

    private void SetOverlayVisible(bool visible)
    {
        if (_deathOverlay != null)
            _deathOverlay.gameObject.SetActive(visible);
    }

    /// <summary>
    /// Programmatically builds a styled restart button on the scene's Canvas.
    /// Falls back to creating a new overlay Canvas if none is found.
    /// </summary>
    private Button CreateRestartButton()
    {
        // Reuse the existing scene canvas when available
        Canvas canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null)
        {
            GameObject canvasGo = new GameObject("RestartCanvas");
            canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 100;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();
        }

        // Full-screen red tint — placed first so the button renders on top
        GameObject overlayGo = new GameObject("DeathOverlay");
        overlayGo.transform.SetParent(canvas.transform, false);
        RectTransform overlayRect = overlayGo.AddComponent<RectTransform>();
        overlayRect.anchorMin = Vector2.zero;
        overlayRect.anchorMax = Vector2.one;
        overlayRect.offsetMin = Vector2.zero;
        overlayRect.offsetMax = Vector2.zero;
        _deathOverlay = overlayGo.AddComponent<Image>();
        _deathOverlay.color = overlayColor;
        _deathOverlay.raycastTarget = false; // let clicks pass through to the button
        overlayGo.SetActive(false);

        // Button root
        GameObject btnGo = new GameObject("RestartButton");
        btnGo.transform.SetParent(canvas.transform, false);

        RectTransform rect = btnGo.AddComponent<RectTransform>();
        rect.sizeDelta = buttonSize;

        // Anchor to top-right
        rect.anchorMin = new Vector2(1f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot     = new Vector2(1f, 1f);
        rect.anchoredPosition = new Vector2(-cornerOffset.x, -cornerOffset.y);

        // Background image
        Image bg = btnGo.AddComponent<Image>();
        bg.color = buttonColor;

        // Button component
        Button btn = btnGo.AddComponent<Button>();
        ColorBlock cb = btn.colors;
        cb.highlightedColor = new Color(1f, 0.35f, 0.35f, 1f);
        cb.pressedColor     = new Color(0.6f, 0.1f, 0.1f, 1f);
        btn.colors = cb;
        btn.targetGraphic = bg;

        // Label
        GameObject textGo = new GameObject("Label");
        textGo.transform.SetParent(btnGo.transform, false);

        RectTransform textRect = textGo.AddComponent<RectTransform>();
        textRect.anchorMin        = Vector2.zero;
        textRect.anchorMax        = Vector2.one;
        textRect.offsetMin        = Vector2.zero;
        textRect.offsetMax        = Vector2.zero;

        TextMeshProUGUI label = textGo.AddComponent<TextMeshProUGUI>();
        label.text      = "RESTART";
        label.color     = labelColor;
        label.fontSize  = 20f;
        label.fontStyle = FontStyles.Bold;
        label.alignment = TextAlignmentOptions.Center;

        Debug.Log("[RestartManager] Restart button created automatically.");
        return btn;
    }
}
