import cv2
from skimage import feature
import joblib
from datetime import datetime as dt
import train
import statistics 
import pandas as pd
import face_recognition
from PIL import Image
import os

# Function to recognize faces in a given image
def recognize_face(image_path, svm_model):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print("No faces found in the image:", image_path)
        return None

    # Assume there is only one face in the image for simplicity
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    face_image = cv2.resize(face_image, (112, 92))

    feat, _ = feature.hog(face_image, orientations=8, pixels_per_cell=(16, 16),
                          visualize=True, block_norm='L2-Hys', cells_per_block=(1, 1))
    
    val1 = svm_model.predict(feat.reshape(1, -1))
    return int(val1)

# Function to mark attendance for a recognized face
def mark_attendance(got_label):
    df = pd.read_csv("./target.csv").drop("Unnamed: 0", axis=1)
    got_name = str(df.loc[int(got_label), "name"])
    
    df_attendance = pd.read_csv("./Attendance_record/" + got_name + ".csv").drop("Unnamed: 0", axis=1)
    length = len(df_attendance)

    # Marking the attendance
    if length == 0:
        df_attendance.loc[0] = [dt.now().strftime("%d/%m/%y"), dt.now().strftime("%H:%M"), "Entered"]
    elif df_attendance.loc[length - 1, "Attendance"] == "Entered":
        df_attendance.loc[length] = [dt.now().strftime("%d/%m/%y"), dt.now().strftime("%H:%M"), "Exit"]
    elif df_attendance.loc[length - 1, "Attendance"] == "Exit":
        df_attendance.loc[length] = [dt.now().strftime("%d/%m/%y"), dt.now().strftime("%H:%M"), "Entered"]

    df_attendance.to_csv("./Attendance_record/" + got_name + ".csv")
    print("Attendance Marked for", got_name)

def main():
    svm_model = joblib.load('svm_face_train_modelnew.pkl')

    # Path to the folder containing individual face images
    folder_path = 'test1'

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            got_label = recognize_face(image_path, svm_model)

            if got_label is not None:
                mark_attendance(got_label)

if _name_ == "_main_":
    main()