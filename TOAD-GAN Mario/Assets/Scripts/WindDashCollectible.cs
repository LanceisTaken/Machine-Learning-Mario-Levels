using UnityEngine;

/// <summary>
/// Wind / Air Dash collectible item.
/// Same slide-and-bounce movement as a Super Mushroom.
/// On collection: grants Mario 3 dash charges.
/// </summary>
public class WindDashCollectible : PowerUpItem
{
    [Header("Dash Grant")]
    [Tooltip("Number of dash charges granted on collection.")]
    public int dashCharges = 3;

    protected override void Collect(PlayerController player)
    {
        player.GrantDash(dashCharges);
        UIPopup.Show($"DASH ×{dashCharges}", transform.position, new Color(0.4f, 0.8f, 1f));
    }
}
