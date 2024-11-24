using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float moveSpeed = 5f;
    public float rotateSpeed = 100f;
    public float zoomSpeed = 500f;

    void Start()
    {
        transform.position = new Vector3(0f, 90f, 10f);
        transform.rotation = Quaternion.Euler(90f, 0f, 45f);
    }

    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector3 movement = new Vector3(horizontal, 0, vertical) * moveSpeed * Time.deltaTime;
        transform.Translate(movement);

        if (Input.GetKey(KeyCode.Q))
            transform.Translate(Vector3.down * moveSpeed * Time.deltaTime);
        if (Input.GetKey(KeyCode.E))
            transform.Translate(Vector3.up * moveSpeed * Time.deltaTime);

        float scroll = Input.GetAxis("Mouse ScrollWheel");
        transform.Translate(Vector3.forward * scroll * zoomSpeed * Time.deltaTime);

        if (Input.GetMouseButton(1))
        {
            float mouseX = Input.GetAxis("Mouse X");
            float mouseY = Input.GetAxis("Mouse Y");

            transform.RotateAround(transform.position, Vector3.up, mouseX * rotateSpeed * Time.deltaTime);
            transform.RotateAround(transform.position, transform.right, -mouseY * rotateSpeed * Time.deltaTime);
        }
    }
}