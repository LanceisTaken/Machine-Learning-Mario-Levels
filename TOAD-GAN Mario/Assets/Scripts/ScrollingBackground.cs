using UnityEngine;

/// <summary>
/// Infinite looping background that scrolls only when the player moves.
/// The UV offset is driven purely by the player's horizontal movement —
/// no movement = no scroll, move right = scroll right, move left = scroll left.
///
/// SETUP STEPS:
///   1. Create a Material (Assets > Create > Material)
///      - Shader: Unlit/Texture
///      - Texture: fyp_background_HD  (Wrap Mode must be "Repeat" in its import settings)
///   2. Create a 3D Quad (GameObject > 3D Object > Quad), rename it "ScrollingBackground"
///      - Assign the Material to its MeshRenderer
///      - DRAG IT ONTO "Main Camera" in the Hierarchy (child of camera)
///      - Set local Position to (0, 0, 15)
///      - Leave Scale at (1, 1, 1) — script auto-sizes it
///   3. Attach this script to the Quad.
///   4. Assign the Player transform in the Inspector.
/// </summary>
[RequireComponent(typeof(Renderer))]
public class ScrollingBackground : MonoBehaviour
{
    [Header("Player")]
    [Tooltip("Drag the Player GameObject here. The background scrolls based on player movement.")]
    public Transform player;

    [Header("Scroll")]
    [Tooltip("How many UV units shift per world unit the player moves. Lower = slower background scroll.")]
    public float scrollSensitivity = 0.005f;

    [Header("Camera")]
    [Tooltip("Leave null to auto-find Camera.main.")]
    public Camera cam;

    private Material _mat;
    private float    _lastPlayerX;
    private float    _uvOffsetX;

    private void Awake()
    {
        _mat = GetComponent<Renderer>().material;

        if (cam == null)
            cam = Camera.main;

        // Auto-find player by tag if not assigned in Inspector
        if (player == null)
        {
            GameObject p = GameObject.FindGameObjectWithTag("Player");
            if (p != null) player = p.transform;
        }

        if (player != null)
            _lastPlayerX = player.position.x;

        FitToCamera();
    }

    private void FitToCamera()
    {
        if (cam == null) return;

        float distZ = Mathf.Abs(transform.localPosition.z);
        float height, width;

        if (cam.orthographic)
        {
            height = cam.orthographicSize * 2f;
            width  = height * cam.aspect;
        }
        else
        {
            height = 2f * distZ * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            width  = height * cam.aspect;
        }

        transform.localScale = new Vector3(width * 1.05f, height * 1.05f, 1f);
    }

    private void LateUpdate()
    {
        if (_mat == null || player == null) return;

        float playerDeltaX = player.position.x - _lastPlayerX;
        _lastPlayerX = player.position.x;

        // Only scroll when player actually moved
        if (playerDeltaX != 0f)
        {
            _uvOffsetX += playerDeltaX * scrollSensitivity;
            _uvOffsetX  = Mathf.Repeat(_uvOffsetX, 1f);
            _mat.mainTextureOffset = new Vector2(_uvOffsetX, 0f);
        }
    }

    private void OnDestroy()
    {
        if (_mat != null)
            Destroy(_mat);
    }
}
