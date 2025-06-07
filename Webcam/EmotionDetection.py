import os
import pickle
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def clearDataset():
    root = os.getcwd()
    datasetPath = os.path.join(root, 'Dataset', 'emotion')
    TARGET_SIZE = (612, 408)
    # Walk through all folders from datasetPath and process image files.
    for dirPath, dirNames, fileNames in os.walk(datasetPath):
        for fileName in fileNames:
            filePath = os.path.join(dirPath, fileName)
            try:
                with Image.open(filePath) as img:
                    if img.size != TARGET_SIZE:

                        resized_img = img.resize(TARGET_SIZE, Image.ANTIALIAS)
                        resized_img.save(filePath)
            except:
                # Only try to remove if file still exists
                if os.path.exists(filePath):
                    os.remove(filePath)


def getFaceLandmarks(img, draw=False):

    faceMesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    results = faceMesh.process(img)

    landmarks = []

    if results.multi_face_landmarks:

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            cv.imshow('img', img)
            cv.waitKey(1)

        lsSingleFace = results.multi_face_landmarks[0].landmark
        xs = []
        ys = []
        zs = []

        # Coordinates
        for idx in lsSingleFace:
            xs.append(idx.x)
            ys.append(idx.y)
            zs.append(idx.z)
        # Normalizing Coordinates
        for j in range(len(xs)):
            landmarks.append(xs[j] - min(xs))
            landmarks.append(ys[j] - min(ys))
            landmarks.append(zs[j] - min(zs))

    return landmarks


def saveFaceLandmarks():
    root = os.getcwd()
    datasetPath = os.path.join(root, 'Dataset', 'emotion')

    output = []

    # Walk through all folders from datasetPath and process image files.
    for dirPath, dirNames, fileNames in os.walk(datasetPath):
        for fileName in fileNames:
            filePath = os.path.join(dirPath, fileName)

            img = cv.imread(filePath)
            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            faceLandmarks = getFaceLandmarks(imgRGB)

            if len(faceLandmarks) == 1404:
                emotion = os.path.basename(dirPath)  # TODO Convert it to Integer
                if emotion == 'happy':
                    emotion_indx = 0
                else:
                    emotion_indx = 1
                faceLandmarks.append(emotion_indx)
                output.append(faceLandmarks)

        np.savetxt('data.txt', np.asarray(output))


def trainModel():
    # Load data from the text file
    data_file = "data.txt"
    data = np.loadtxt(data_file)

    # Split data into features (x) and labels (y)
    x = data[:, :-1]
    y = data[:, -1]

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y  # keeps the same class proportions in both sets
                                                        )
    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Train the classifier on the training data
    rf_classifier.fit(X_train, Y_train)

    # Make prediction on the test data
    Y_pred = rf_classifier.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy is {accuracy}')

    # Save the model
    with open('./model', 'wb') as f:
        pickle.dump(rf_classifier, f)


def webcamTest():

    # Read Webcam
    cap = cv.VideoCapture(0)

    # Load the model
    with open('./model', 'rb') as f:
        model = pickle.load(f)

    emotions = ['Happy', 'Sad']

    while cap.isOpened():

        _, frame = cap.read()

        # Get Landmarks
        landmarks = getFaceLandmarks(frame, draw=False)

        # Run Model
        output = model.predict([landmarks])

        # Show Result
        cv.putText(frame,
                   emotions[int(output[0])],
                   (10, frame.shape[0] - 1),
                   cv.FONT_HERSHEY_SIMPLEX,
                   3,
                   (0, 255, 0),
                   5)

        cv.imshow('Emotion Detection', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # clearDataset()
    # saveFaceLandmarks()
    # trainModel()
    webcamTest()
