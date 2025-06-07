import os
import shutil
import cv2 as cv
from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def rawDataToDataset():
    root = os.getcwd()
    datasetPath = os.path.join(root, 'Dataset', 'weather')

    # Create destination folders if they don't exist
    categories = ['cloudy', 'shine', 'rain', 'sunrise']
    for category in categories:
        category_path = os.path.join(datasetPath, category)
        os.makedirs(category_path, exist_ok=True)

    # Walk through all folders from datasetPath and process image files.
    for dirPath, dirNames, fileNames in os.walk(datasetPath):
        for fileName in fileNames:
            filePath = os.path.join(dirPath, fileName)

            for category in categories:
                if fileName.lower().startswith(category):
                    destPath = os.path.join(dirPath, category, fileName)
                    # Move each data to its corresponding folder
                    shutil.move(filePath, destPath)
                    print(f"Moved {fileName} to {category}/")


def prepareData():

    # Feature Extraction Model
    img2vec = Img2Vec()
    features = []
    labels = []

    root = os.getcwd()
    datasetPath = os.path.join(root, 'Dataset', 'weather')

    # Walk through all folders from datasetPath and process image files.
    for dirPath, dirNames, fileNames in os.walk(datasetPath):
        for fileName in fileNames:
            filePath = os.path.join(dirPath, fileName)

            img = Image.open(filePath).convert('RGB')

            imgFeatures = img2vec.get_vec(img)
            features.append(imgFeatures)
            labels.append(os.path.basename(dirPath))

    return features, labels


def trainModel():

    x, y = prepareData()

    # Split train and validation data
    X_train, X_test, Y_train, Y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y)

    # Initialize the model
    model = RandomForestClassifier()

    # Train the classifier on the training data
    model.fit(X_train, Y_train)

    # Make prediction on the test data
    Y_pred = model.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy is {accuracy}')

    # Save the model
    with open('./model', 'wb') as f:
        pickle.dump(model, f)


def test():
    # Initialize the feature extraction model
    img2vec = Img2Vec()

    # Test image
    testImgPath = './Dataset/weather/sunrise/sunrise5.jpg'
    img = Image.open(testImgPath)

    # load the model
    with open('./model', 'rb') as f:
        model = pickle.load(f)

    features = img2vec.get_vec(img)
    predict = model.predict([features])

    print(predict)


if __name__ == '__main__':
    # rawDataToDataset()
    # prepareData()
    # trainModel()
    test()
