import face_recognition
from PIL import Image

# Load your image
image_path = '1.jpg'
image = face_recognition.load_image_file(image_path)

# Find all face locations
face_locations = face_recognition.face_locations(image)

# Iterate over each face found
for i, face_location in enumerate(face_locations):
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)

    # Save the face image
    pil_image.save(f"face_{i}.jpg")