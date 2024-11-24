using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float moveSpeed = 10f;
    public float rotateSpeed = 100f;
    public float zoomSpeed = 500f;

    void Start()
    {
        // Set initial camera position and rotation
        transform.position = new Vector3(-40f, 15f, 0f);  // Adjust these values to your preferred starting position
        transform.rotation = Quaternion.Euler(45f, 90f, 0f);  // Adjust these angles to your preferred starting view
    }

    void Update()
    {
        // WASD or Arrow keys for movement
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector3 movement = new Vector3(horizontal, 0, vertical) * moveSpeed * Time.deltaTime;
        transform.Translate(movement);

        // Q/E for up/down movement
        if (Input.GetKey(KeyCode.Q))
            transform.Translate(Vector3.down * moveSpeed * Time.deltaTime);
        if (Input.GetKey(KeyCode.E))
            transform.Translate(Vector3.up * moveSpeed * Time.deltaTime);

        // Mouse wheel for zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        transform.Translate(Vector3.forward * scroll * zoomSpeed * Time.deltaTime);

        // Hold right mouse button to rotate
        if (Input.GetMouseButton(1))
        {
            float mouseX = Input.GetAxis("Mouse X");
            float mouseY = Input.GetAxis("Mouse Y");

            transform.RotateAround(transform.position, Vector3.up, mouseX * rotateSpeed * Time.deltaTime);
            transform.RotateAround(transform.position, transform.right, -mouseY * rotateSpeed * Time.deltaTime);
        }
    }
}