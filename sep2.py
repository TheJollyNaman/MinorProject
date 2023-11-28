import cv2

def capture_and_save_frame(camera_index=0, output_path="captured_frame.jpg"):
    """
    Capture a single frame from a video capture and save it as a JPEG image.

    Parameters:
    - camera_index (int): Index of the camera (default is 0 for the default camera).
    - output_path (str): Path to save the captured frame as a JPEG image.

    Returns:
    - frame (numpy.ndarray): Captured frame as a NumPy array.
    """
    vid = cv2.VideoCapture(0)
    while True:
        # Capture a single frame
        ret, frame = vid.read()

        # Display the frame with recognized faces
        cv2.imshow('Recognized Faces', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the captured frame as a JPEG image
    cv2.imwrite(output_path, frame)

    # Release the video capture
    vid.release()
    cv2.destroyAllWindows()
    return frame

if __name__ == "__main__":
    # Example: Capture a frame from the default camera (index 0) and save it as a JPEG image
    captured_frame = capture_and_save_frame(output_path="captured_frame.jpg")

    # Display the captured frame (optional)
    cv2.imshow('Captured Frame', captured_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Captured frame saved as 'captured_frame.jpg'")
