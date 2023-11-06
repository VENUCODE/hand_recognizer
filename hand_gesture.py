import mediapipe as mp
import cv2 as cv
#handland mark detection model
#palm detection model
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mphands=mp.solutions.hands #for tracking the hand

#accessing the webcam using the webacm
capture=cv.VideoCapture(0)
while True:
    data,image=capture.read()
    
    #flipping the image
    image=cv.cvtColor(cv.flip(image,1),cv.COLOR_BGR2RGB)#selfie view
    results=mphands.Hands().process(image) #storing the video image results
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS)
    cv.imshow("hand mark",image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break