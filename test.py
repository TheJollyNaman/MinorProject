import cv2
from skimage import feature
import joblib
from datetime import datetime as dt
import train
import statistics
import pandas as pd

def recognize_face(image_path):
    # Load the pre-trained SVM model
    svm_model = joblib.load('svm_face_train_modelnew.pkl')

    # Read the input image
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (640, 480))

    # Convert the image to grayscale
    im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar cascade for frontal face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(im1, 1.3, 5)

    recognized_labels = []

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 4)

        # Extract the face region
        im_f = im1[y:y + h, x:x + w]
        im_f = cv2.resize(im_f, (112, 92))

        # Extract HOG features
        feat, hog_image = feature.hog(im_f, orientations=8, pixels_per_cell=(16, 16),
                                      visualize=True, block_norm='L2-Hys',
                                      cells_per_block=(1, 1))

        # Make prediction using the SVM model
        val1 = svm_model.predict(feat.reshape(1, -1))
        str1 = str(val1)

        # Display the result on the image
        cv2.putText(frame, str1, (x, y), cv2.FONT_ITALIC, 1, (255, 0, 255), 2, cv2.LINE_AA)

        recognized_labels.append(int(val1))

    # Display the image with recognized faces
    cv2.imshow('Recognized Faces', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Handle the case when no faces are detected
    if not recognized_labels:
        print("No faces detected in the image.")
        return None

    # Return the recognized label based on the mode of detected labels
    recognized_label = statistics.mode(recognized_labels)
    return recognized_label

if __name__ == "__main__":
    image_path = '2.png'  # Replace with the path to your test image
    recognized_label = recognize_face(image_path)

    if recognized_label is not None:
        # Load the CSV file with target names
        df = pd.read_csv("./target.csv").drop("Unnamed: 0", axis=1)

        # Get the recognized name based on the label
        got_name = str(df.loc[int(recognized_label), "name"])
        print(f"Recognized Name: {got_name}")
