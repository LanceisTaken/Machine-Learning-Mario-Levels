using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -10);
    public float smoothSpeed = 5f;

    private void LateUpdate()
    {
        if (target == null) return;
        Vector3 desired = target.position + offset;
        transform.position = Vector3.Lerp(
            transform.position, desired, smoothSpeed * Time.deltaTime);
    }
}