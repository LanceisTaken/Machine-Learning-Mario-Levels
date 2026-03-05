using UnityEngine;

public class EnemyPatrol : MonoBehaviour
{
    public float speed = 2f;
    private Rigidbody2D _rb;
    private int _direction = -1; // Start moving left

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
    }

    private void FixedUpdate()
    {
        _rb.linearVelocity = new Vector2(_direction * speed, _rb.linearVelocity.y);
    }

    private void OnCollisionEnter2D(Collision2D col)
{
    if (col.gameObject.CompareTag("Player"))
    {
        // Player landed on top → kill enemy
        if (col.GetContact(0).normal.y < -0.5f)
        {
            Destroy(gameObject);
            // Give the player a small bounce
            var playerRb = col.gameObject.GetComponent<Rigidbody2D>();
            playerRb.linearVelocity = new Vector2(
                playerRb.linearVelocity.x, 8f);
        }
        else
        {
            // Enemy hit player from the side → damage player
            Debug.Log("Player hit!");
        }
    }
    else
    {
        // Reverse direction when hitting a wall
        _direction *= -1;
    }
}
}