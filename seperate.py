import os
from datetime import datetime
import cv2
import face_recognition
from PIL import Image
 

def save_single_frame_from_camera():
    # Open the camera
    vid = cv2.VideoCapture(0)

    while True:
        # Capture a single frame
        ret, frame = vid.read()

        # Display the frame with recognized faces
        cv2.imshow('Recognized Faces', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Create a new folder with the current date and time
    image_path = 'captured_frame.jpg'
    cv2.imwrite(image_path, frame)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"detected_faces_{current_datetime}"
    cv2.imshow('Recognized Faces', frame)
    # Find all face locations in the captured frame
    
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    os.chdir("Faces")
    os.makedirs(output_folder)
    print(face_locations)
    # Iterate over each face found
    for i, face_location in enumerate(face_locations):
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

        # Access the actual face itself
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        
        # Save the face image in the new folder
        face_filename = f"face_{i}.jpg"
        face_filepath = os.path.join(output_folder, face_filename)
        pil_image.save(face_filepath)
        print(f"Saved face {i} to {face_filepath}")

    # Release the camera and close the OpenCV window
    vid.release()
    cv2.destroyAllWindows()
    current_directory = os.getcwd()
    parent_directory = os.path.join(current_directory, '..')
    os.chdir(parent_directory)
    return output_folder

