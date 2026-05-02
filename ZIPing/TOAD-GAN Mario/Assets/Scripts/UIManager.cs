using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UIManager : MonoBehaviour
{
    [Header("Health UI")]
    [Tooltip("Heart sprite shown beside the count. Put this object and Health Text as children of the same row with a Horizontal Layout Group (icon left, text right).")]
    public Image heartIcon;
    [Tooltip("Shows remaining health, e.g. format x{0} → x3")]
    public TextMeshProUGUI healthText;
    public string healthFormat = "x{0}";

    [Header("Ability UI")]
    public TextMeshProUGUI dashText;
    public TextMeshProUGUI jumpText;

    public void UpdateHealth(int currentHealth)
    {
        currentHealth = Mathf.Max(0, currentHealth);
        if (healthText != null)
            healthText.text = string.Format(healthFormat, currentHealth);
    }

    public void UpdateDashes(int dashesLeft)
    {
        if (dashText == null) return;
        dashText.text = "Dash: " + Mathf.Max(0, dashesLeft);
    }

    public void UpdateJumps(int jumpsLeft)
    {
        if (jumpText == null) return;
        jumpText.text = "Jump: " + Mathf.Max(0, jumpsLeft);
    }
}
