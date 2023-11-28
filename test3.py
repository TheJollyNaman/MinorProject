import cv2
from skimage import feature
import joblib
import os
import pandas as pd
import numpy as np
from statistics import median
from datetime import datetime as dt
import seperate

def recognize_faces_in_folder(folder_path, threshold=0.5):
    # Load the pre-trained SVM model
    svm_model = joblib.load('svm_face_train_modelnew.pkl')

    # Haar cascade for frontal face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load the CSV file with target names
    df_target = pd.read_csv("./target.csv").drop("Unnamed: 0", axis=1)

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
            frame = face_cascade.detectMultiScale(frame, 1.3, 5)
            
            im_f = im1
            im_f = cv2.resize(im_f, (112, 92))
            # Extract HOG features
            feat, hog_image = feature.hog(im_f, orientations=8, pixels_per_cell=(16, 16),
                                            visualize=True, block_norm='L2-Hys',
                                            cells_per_block=(1, 1))

            # Make prediction using the SVM model
            confidence = svm_model.decision_function(feat.reshape(1, -1))
            predicted_class_index = np.argmax(confidence)
            # mean(confidence[0])
            print(confidence[0])
            if confidence[0, predicted_class_index] > threshold:
                recognized_id = svm_model.predict(feat.reshape(1, -1))
                recognized_name = df_target.loc[int(recognized_id), "name"]
            else:
                recognized_name = "Unknown"

            recognized_names.append(recognized_name)

            # Display the result on the image
            # cv2.putText(frame, recognized_name, (x, y), cv2.FONT_ITALIC, 1, (255, 0, 255), 2, cv2.LINE_AA)
                

            # Display the image with recognized faces
            # cv2.imshow('Recognized Faces', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # Print the recognized names in the terminal
    print("Recognized Names:")
    for name in recognized_names:
        print(name)
        df = pd.read_csv("./Attendence_record/"+name+".csv").drop("Unnamed: 0",axis=1)
        length = len(df)
        #marking the attandence
        # if length==0:
        #     df.loc[0]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Entered"]
        # elif (df.loc[length-1,"Attendence"]=="Entered"):
        #     df.loc[length]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Exit"]
        # elif (df.loc[length-1,"Attendence"]=="Exit"):
        #     df.loc[length]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Entered"]
        
        df.loc[length]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Entered"]
        df.to_csv("./Attendence_record/"+name+".csv")
        print("Attendence Marked")

def main():
    outputfolder = seperate.save_single_frame_from_camera()
    folder_path = f"Faces/{outputfolder}"  # Replace with the path to your image folder
    recognize_faces_in_folder(folder_path, threshold=0.5)
