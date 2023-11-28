import cv2
from skimage import feature
import joblib
from datetime import datetime as dt
import train
import statistics
import pandas as pd
import os

def recognize_faces_in_folder(folder_path):
    # Load the pre-trained SVM model
    svm_model = joblib.load('svm_face_train_modelnew.pkl')

    # Haar cascade for frontal face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    recognized_names = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)

            # Convert the image to grayscale
            im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(im1, 1.3, 5)

            for x, y, w, h in faces:
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

                # Store the recognized name in the list
                recognized_names.append(str1)

            # Display the image with recognized faces
            cv2.imshow('Recognized Faces', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Print the recognized names in the terminal
    print("Recognized ID:")
    for name in recognized_names:
        print(name)

if __name__ == "__main__":
    folder_path = 'test1'  # Replace with the path to your image folder
    recognize_faces_in_folder(folder_path)
