import cv2 as cv
import mediapipe as mp


def FaceAnonymizer():

    cap = cv.VideoCapture(0)

    mp_face_detection = mp.solutions.face_detection

    # model_selection: Use 0 to select a short-range model that works best for faces within 2 meters from camera
    #                  Use 1 to select a full-range  model that works best for faces within 5 meters from camera
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:

        while cap.isOpened():

            _, frame = cap.read()

            out = face_detection.process(frame)

            if out.detections is not None:
                for detection in out.detections:
                    location_data = detection.location_data
                    bbox = location_data.relative_bounding_box

                    # Convert Coordinates
                    h_frame, w_frame, _ = frame.shape
                    x1 = int(bbox.xmin * w_frame)
                    y1 = int(bbox.ymin * h_frame)
                    x2 = int((bbox.xmin + bbox.width) * w_frame)
                    y2 = int((bbox.ymin + bbox.height) * h_frame)

                    # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

                    # Blur The Face
                    frame[y1:y2, x1:x2] = cv.blur(frame[y1:y2, x1:x2], (50, 50))

            cv.imshow('Face Detection', frame)

            if cv.waitKey(1) == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    FaceAnonymizer()
