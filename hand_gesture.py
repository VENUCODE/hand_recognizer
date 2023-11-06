import mediapipe as mp
import cv2 as cv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

# Accessing the webcam using the webcam
capture = cv.VideoCapture(0)

with mphands.Hands() as hands:
    finger_tip_path = []

    while True:
        data, image = capture.read()

        # Flipping the image
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)  # Selfie view
        results = hands.process(image)  # Storing the video image results

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]

                # Draw the index finger tip as a point
                image_height, image_width, _ = image.shape
                x, y = int(index_finger_tip.x * image_width), int(index_finger_tip.y * image_height)
                cv.circle(image, (x, y), 10, (0, 0, 255), -1)  # Draw a red circle at the index finger tip

                # Add the current position to the path
                finger_tip_path.append((x, y))

                if len(finger_tip_path) > 1:
                    # Draw a curve connecting the positions in the path
                    cv.polylines(image, [np.array(finger_tip_path)], isClosed=False, color=(0, 0, 255), thickness=2)
        else:
            finger_tip_path=[]
        cv.imshow("hand mark", image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
