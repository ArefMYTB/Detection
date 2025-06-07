import cv2 as cv
import pytesseract
from easyocr import Reader


def TextDetection(model):

    cap = cv.VideoCapture(0)

    while cap.isOpened():

        _, frame = cap.read()

        if model == 'Pytesseract':
            # Pytesseract
            # Camera usually has low quality so
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh, lang='eng')
            print(text)
        else:
            # EasyOCR
            text = ''
            reader = Reader(['en'])
            results = reader.readtext(frame)
            for result in results:
                text = text + result[1] + ' '
            text = text.strip()  # Don't care about last space -- text[:-1]
            print(text)

        cv.imshow('Text Detection', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    TextDetection(model='EasyOCR')  # or Pytesseract
