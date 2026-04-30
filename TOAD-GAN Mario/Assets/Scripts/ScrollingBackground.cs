using UnityEngine;

/// <summary>
/// Infinite looping background via UV-offset scrolling on a Quad with a Repeat-wrapped material.
///
/// SETUP STEPS (do once in the Editor):
///   1. Create a new Material: Assets > Create > Material
///      - Shader: "Unlit/Texture"  (or Sprites/Default won't tile — use Unlit/Texture)
///      - Set the Albedo/Base Map to your fyp_background texture
///      - In the Texture import settings, set Wrap Mode to "Repeat"
///   2. In the Scene, create a 3D Quad: GameObject > 3D Object > Quad
///      - Rename it "ScrollingBackground"
///      - Assign your new Material to its MeshRenderer
///      - Scale it to cover the full camera view, e.g. (camera ortho size * aspect * 2) wide
///        and (camera ortho size * 2) tall. A good starting scale: X=40, Y=12, Z=1
///      - Position Z should be behind everything, e.g. Z = 10 (or sort by layer if 2D)
///      - Rotate X by 0 (face the camera; if it disappears, rotate X by 180)
///   3. Attach this script to the Quad.
///   4. Set "Parallax Factor" in the Inspector (0 = no movement, 1 = moves with camera, 0.2 = subtle depth)
///   5. Set "Scroll Speed" for auto-scroll (set to 0 to disable auto-scroll and only use parallax).
///
/// ALTERNATIVELY — if you want a pure auto-scroll (not camera-relative):
///   Set parallaxFactor = 0 and scrollSpeed to something like 0.05.
/// </summary>
[RequireComponent(typeof(Renderer))]
public class ScrollingBackground : MonoBehaviour
{
    [Header("Parallax (camera-relative)")]
    [Tooltip("How much the background moves relative to the camera. 0 = static, 0.2 = subtle depth, 1 = moves with camera.")]
    [Range(0f, 1f)]
    public float parallaxFactor = 0.2f;

    [Header("Auto-Scroll")]
    [Tooltip("Constant UV scroll speed (units/second). Set to 0 to rely on parallax only.")]
    public float scrollSpeed = 0.03f;

    [Header("Camera")]
    [Tooltip("Leave null to auto-find Camera.main.")]
    public Camera cam;

    // Internal state
    private Material _mat;
    private float    _lastCamX;
    private float    _uvOffsetX;

    private void Awake()
    {
        // Use an instance material so we don't modify the shared asset
        _mat = GetComponent<Renderer>().material;

        if (cam == null)
            cam = Camera.main;

        if (cam != null)
            _lastCamX = cam.transform.position.x;
    }

    private void LateUpdate()
    {
        if (_mat == null) return;

        // ── Camera-relative parallax ───────────────────────────────────────
        if (cam != null)
        {
            float camDeltaX = cam.transform.position.x - _lastCamX;
            _uvOffsetX  += camDeltaX * parallaxFactor * 0.01f; // scale to UV space
            _lastCamX    = cam.transform.position.x;

            // Keep background quad centred on camera so it always fills the view
            Vector3 pos = transform.position;
            pos.x = cam.transform.position.x;
            pos.y = cam.transform.position.y;
            transform.position = pos;
        }

        // ── Constant auto-scroll ───────────────────────────────────────────
        _uvOffsetX += scrollSpeed * Time.deltaTime;

        // Keep value in [0,1] to avoid float precision drift over long sessions
        _uvOffsetX = Mathf.Repeat(_uvOffsetX, 1f);

        _mat.mainTextureOffset = new Vector2(_uvOffsetX, 0f);
    }

    private void OnDestroy()
    {
        // Clean up the instance material we created in Awake
        if (_mat != null)
            Destroy(_mat);
    }
}
