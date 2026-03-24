using UnityEngine;

/// <summary>
/// Jump Power-Up collectible item.
/// Same slide-and-bounce movement as other power-ups.
/// On collection: grants Mario double-jump charges.
/// </summary>
public class JumpPowerUpCollectible : PowerUpItem
{
    [Header("Double Jump Grant")]
    [Tooltip("Number of double-jump charges granted on collection.")]
    public int jumpCharges = 2;

    protected override void Collect(PlayerController player)
    {
        player.GrantDoubleJump(jumpCharges);
        UIPopup.Show($"JUMP ×{jumpCharges}", transform.position, new Color(0.6f, 1f, 0.6f));
    }
}
