from collections import defaultdict
import math,os,pickle,random,cv2,face_recognition,firebase_admin,winsound
import numpy as np
import cvzone as cz
from cvzone.FaceDetectionModule import FaceDetector
from firebase_admin import credentials, storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from ultralytics import YOLO
from datetime import datetime
from firebase_admin import db

# YOLO Object Detection Setup
cap = cv2.VideoCapture(0)
object_counts = defaultdict(int)
height, width = 300, 400
random_color = tuple(np.random.randint(0, 255, size=3).tolist())
color_wall = (0, 255, 0)
model = YOLO('yolov8n.pt')
random_number = random.randint(1000, 9999)

# List of object classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

previous_points = []
count = 0

# Firebase Setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceface-f5d7f-default-rtdb.firebaseio.com/",
    'storageBucket': 'faceface-f5d7f.appspot.com'
})
bucket = storage.bucket()

face_detector = FaceDetector(minDetectionCon=0.05, modelSelection=0)

# Google Drive API setup
#SCOPES = ['https://www.googleapis.com/auth/drive']
#SERVICE_ACCOUNT_FILE = 'fordriveapi.json'  # Path to the JSON file you downloaded

#credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#drive_service = build('drive', 'v3', credentials=credentials)

# Function to upload file to Google Drive
"""
def upload_to_drive(file_path, folder_id=None):
    file_metadata = {'name': os.path.basename(file_path)}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: {file.get("id")}')
"""

def fetch_student_image(student_id):
    """
    Fetch student image from Firebase storage.
    """
    blob = bucket.get_blob(f'Images/{student_id}.jpg')
    if blob:
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        img_student = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return img_student
    else:
        print(f"Image for student ID {student_id} not found.")
        return None


def main():
    global count, color_wall

    # Load known encodings and student IDs
    with open('Encoding.p', "rb") as file:
        known_encodings, student_ids = pickle.load(file)

    output_file = 'output.mp4'
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))

    while True:
        # YOLO Object Detection
        ret, frame = cap.read()
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        forinfo = np.zeros((height, width, 3), dtype=np.uint8)

        if not ret:
            break
        now = datetime.now()

        # Format the date and time
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

        # Print the formatted date and time
        cz.putTextRect(frame, f"Current date and time: {formatted_now}", (50, 50), 1, 2,)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (900, 150), color_wall, -1)

        res = model(frame, stream=True)
        count = 0
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cx, cy = (x1 + w // 2), (y1 + h // 2)
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                if classes[cls] == 'person' and 10 < cx < 300 and 10 < cy < 300:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                    previous_points.append((cx, cy))
                    count += 1
                    color_wall = (0, 0, 255)
                    cz.putTextRect(forinfo, f'count person :{count}', (50, 50), 1, 2)
                elif classes[cls] == 'cat' or classes[cls] == 'dog' and 10 < cx < 300 and 10 < cy < 300:
                    winsound.Beep(500,200)
                elif classes[cls] != 'person':
                    color_wall = (0, 255, 0)

        if len(previous_points) > 1:
            for i in range(len(previous_points) - 1):
                cv2.circle(frame, previous_points[i], 1, (0, 255, 0), -1)
                cv2.circle(black_image, previous_points[i], 1, random_color, -1)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Face Recognition
        img_face, bboxs = face_detector.findFaces(frame)
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox['bbox']
                face = img_face[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 1)
                if face.size == 0:
                    print("Warning: Detected face region is empty.")
                    continue
                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    print(f"Error converting face to RGB: {e}")
                    continue
                encodings_cur_frame = face_recognition.face_encodings(face_rgb)
                if encodings_cur_frame:
                    for encoding in encodings_cur_frame:
                        matches = face_recognition.compare_faces(known_encodings, encoding)
                        face_distances = face_recognition.face_distance(known_encodings, encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            student_id = student_ids[best_match_index]
                            print(f"Match found with ID: {student_id}")
                            student_img = fetch_student_image(student_id)
                            cz.putTextRect(img_face, 'I know this face', (x, y - 10), 1, 2)
                            print('------------------------------------------'
                                  '---------------------------------------')
                        else:
                            while True:
                                cz.putTextRect(img_face, 'I dont know this face', (x, y - 10), 1, 2)
                                print('===============================')
                                print('send mesage there is some person i dont him')
                                print('===============================')
                                cv2.imwrite(f"faceIDK/{random_number}.jpg", face)
                                # Reference to the database
                                ref = db.reference(f'Students/{random_number}')

                                # Pushing data to Firebase Realtime Database
                                ref.set({
                                    'i see in': formatted_now
                                })


        cv2.imshow('img_face', img_face)

        # Writing frame to video file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    #upload_to_drive(output_file, folder_id='1hzT-gCZrLCgOzoa8lVN1-d4QUwgZuH6i')  # Replace with your folder ID



if __name__ == "__main__":
    main()
