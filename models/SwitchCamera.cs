using UnityEngine;

public class SwitchCamera : MonoBehaviour
{
    public Camera mainCamera;
    public Camera secondaryCamera;

    private void Start()
    {
        // Initially enable the main camera and disable the secondary camera
        mainCamera.enabled = true;
        secondaryCamera.enabled = false;
    }

    private void Update()
    {
        // Check for input to switch cameras
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // Swap the camera tags
            string tempTag = mainCamera.tag;
            mainCamera.tag = secondaryCamera.tag;
            secondaryCamera.tag = tempTag;

            // Enable/disable cameras based on tags
            mainCamera.enabled = !mainCamera.enabled;
            secondaryCamera.enabled = !secondaryCamera.enabled;

            // Update the mainCamera reference
            if (mainCamera.enabled)
            {
                mainCamera = Camera.main;
            }
        }
    }
}