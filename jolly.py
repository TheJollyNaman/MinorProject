import cv2
import datetime
import face_recognition
import os
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to get the detections from YOLO
def get_detections(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    faces = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                faces.append((x, y, w, h))
    return faces

# Load and encode faces from reference images
known_face_encodings = []
known_face_names = []

for filename in os.listdir('path_to_reference_images'):
    image = face_recognition.load_image_file(f'path_to_reference_images/{filename}')
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(filename.split('.')[0])

# Function to mark attendance
def mark_attendance(name, last_marked, min_interval_seconds=60):
    current_time = datetime.datetime.now()
    if name in last_marked and (current_time - last_marked[name]).total_seconds() < min_interval_seconds:
        return
    last_marked[name] = current_time
    with open('attendance.csv', 'a') as f:
        f.write(f'\n{name},{current_time.strftime("%Y-%m-%d %H:%M:%S")}')

# Main function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    last_marked_attendance = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = get_detections(frame)
        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(face_frame)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                mark_attendance(name, last_marked_attendance)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video('path_to_video.mp4')