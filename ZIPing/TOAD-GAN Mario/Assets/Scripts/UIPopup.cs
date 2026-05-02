using System.Collections;
using UnityEngine;
using TMPro;

/// <summary>
/// Spawns a floating world-space text popup (e.g. "1-UP!", "+1000") that
/// drifts upward and fades out.  Call UIPopup.Show() from anywhere.
/// </summary>
public class UIPopup : MonoBehaviour
{
    // ── Static factory ─────────────────────────────────────────────────────

    private static GameObject _prefab;

    /// <summary>
    /// Display a floating text label at the given world position.
    /// A simple fallback quad-based label is created if no prefab is available.
    /// </summary>
    public static void Show(string text, Vector3 worldPos, Color? color = null)
    {
        // Lazy-create a prefab once
        if (_prefab == null)
        {
            _prefab = CreatePopupPrefab();
        }

        GameObject go = Instantiate(_prefab, worldPos, Quaternion.identity);
        go.SetActive(true); // Triggers Awake() and assigns _label
        UIPopup popup  = go.GetComponent<UIPopup>();
        popup._label.text  = text;
        popup._label.color = color ?? Color.white;
    }

    // ── Instance ───────────────────────────────────────────────────────────

    private TMP_Text _label;

    [SerializeField] private float floatSpeed  = 1.5f;
    [SerializeField] private float lifetime    = 1.2f;

    private void Awake()
    {
        _label = GetComponentInChildren<TMP_Text>();
    }

    private void Start()
    {
        StartCoroutine(FloatAndFade());
    }

    private IEnumerator FloatAndFade()
    {
        float elapsed = 0f;
        Color startColor = _label.color;

        while (elapsed < lifetime)
        {
            float t = elapsed / lifetime;
            transform.position += Vector3.up * floatSpeed * Time.deltaTime;
            _label.color = new Color(startColor.r, startColor.g, startColor.b, 1f - t);
            elapsed += Time.deltaTime;
            yield return null;
        }

        Destroy(gameObject);
    }

    // ── Prefab builder ─────────────────────────────────────────────────────

    private static GameObject CreatePopupPrefab()
    {
        GameObject root = new GameObject("UIPopup");
        root.AddComponent<UIPopup>();

        // Child: TextMeshPro world-space text
        GameObject textGo = new GameObject("Text");
        textGo.transform.SetParent(root.transform, false);

        var tmp = textGo.AddComponent<TextMeshPro>();
        tmp.fontSize           = 4;
        tmp.alignment          = TextAlignmentOptions.Center;
        tmp.color              = Color.white;
        tmp.sortingLayerID     = SortingLayer.NameToID("UI"); // adjust if needed
        tmp.sortingOrder       = 10;
        tmp.text               = "";

        // Make into a proper prefab-like object (just return the prototype)
        root.SetActive(false);   // deactivate prototype
        DontDestroyOnLoad(root);
        return root;
    }
}
