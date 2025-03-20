import os
import cv2
import random
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials, db, storage

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "____",
    'storageBucket': '______'  # Specify your Firebase Storage bucket name here
})

# Function to find matching ID in Firebase based on image filename
def find_id(img_path):
    # Get just the filename without extension
    file_name = os.path.basename(img_path)
    x = os.path.splitext(file_name)[0]
    print(f'Image filename: {x}')

    # Retrieve data from Firebase
    ref = db.reference('Students')
    data = ref.get()
    print(data)
    if data is None:
        print("No data found in Firebase.")
        return None, None

    for key, value in data.items():
        print(f'Checking database key: {key}')
        if x == key:
            datetime_string = value.get('i see in', 'not found')
            print(f'Matched key {key}. Found at {datetime_string}')
            return key, datetime_string  # Returning the found key and datetime string

    print(f'No match found for image {img_path}')
    return None, None

# Path to the folder containing images
foldersPath_ = "faceIDK"
PathList_ = os.listdir(foldersPath_)

for img_name in PathList_:
    img_path = os.path.join(foldersPath_, img_name)
    img1 = cv2.imread(img_path)

    if img1 is not None:
        cv2.imshow('face', img1)
        key, datetime_found = find_id(img_path)
        cv2.waitKey(0)  # Wait indefinitely for user input

        if key is not None:
            inputt = input('Do you know this person and want to add them to the list of known people? (YES/NO): ').strip()
            if inputt.lower() == 'yes':
                random_number = random.randint(1000, 9999)
                cv2.imwrite(f"Images/{random_number}.jpg", img1)
                foldersPath = "Images"
                PathList = os.listdir(foldersPath)
                imgList = []
                studentIds = []
                for path in PathList:
                    imgList.append(cv2.imread(os.path.join(foldersPath, path)))
                    studentIds.append(os.path.splitext(path)[0])
                    fileName = f'{foldersPath}/{path}'
                    bucket = storage.bucket()
                    blob = bucket.blob(fileName)
                    blob.upload_from_filename(fileName)
                print(studentIds)

                def find_encodings(imgs):
                    encodelist = []
                    for img in imgs:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        encode = face_recognition.face_encodings(img)[0]
                        encodelist.append(encode)
                    return encodelist

                encodelist_new = find_encodings(imgList)
                encodelist_with_ids = [encodelist_new, studentIds]
                with open('Encoding.p', "wb") as file:
                    pickle.dump(encodelist_with_ids, file)
            elif inputt.lower() == 'no':
                print('You should call the police (911).')
            else:
                print('Please enter yes or no.')


            folder_path = 'faceIDK'
def delete_all_images(folder_path):
    # Iterate over all the items in the given folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it's a file and if it has a typical image file extension
            if os.path.isfile(file_path) and filename.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                os.unlink(file_path)  # Remove the file
                print(f'Deleted {file_path}')
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
delete_all_images(folder_path)


cv2.destroyAllWindows()  # Close all OpenCV windows at the end




