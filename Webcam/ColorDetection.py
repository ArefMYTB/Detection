import cv2 as cv
import numpy as np
from PIL import Image

# BGR color to HSV lower and upper limits
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


def ColorDetection():

    cap = cv.VideoCapture(0)

    # The Color We Want To Detect
    yellow = [0, 200, 250]  # worked best for my banana 
    lowerLimit, upperLimit = get_limits(color=yellow)

    while cap.isOpened():

        _, frame = cap.read()

        hsvImg = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsvImg, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 1)

        cv.imshow('Color Detection', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    ColorDetection()
