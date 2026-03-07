using UnityEngine;

/// <summary>
/// 1-UP Mushroom power-up.
/// Same slide/bounce movement as Super Mushroom.
/// On collection: grants +1 life (no score).
/// </summary>
public class OneUpMushroom : PowerUpItem
{
    protected override void Collect(PlayerController player)
    {
        if (GameManager.Instance != null)
            GameManager.Instance.AddLife();

        UIPopup.Show("1-UP!", transform.position, new Color(0.4f, 1f, 0.4f));
    }
}
