using UnityEngine;

/// <summary>
/// Super Mushroom power-up.
/// Slides right, bounces off walls, disappears in pits.
/// On collection: grows Mario to Big Mario, awards 1000 points.
/// </summary>
public class SuperMushroom : PowerUpItem
{
    protected override void Collect(PlayerController player)
    {
        player.GrowBig();

        if (GameManager.Instance != null)
            GameManager.Instance.AddScore(1000);

        UIPopup.Show("+1000", transform.position, new Color(1f, 0.9f, 0.2f));
    }
}
