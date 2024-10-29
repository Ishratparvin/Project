#importing libraries
from facial_emotion_recognition import EmotionRecognition
import cv2

#Capturing Video
er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

while True:
    #return 2 values success and frame
    success, frame = cam.read()
    result = er.recognise_emotion(frame,return_type='GBR')

    #result contains 2 values
    if result is not None and len(result) == 2:
        faces, emotions = result

        #zip() function returns a zip object, which is an iterator of tuples where the
        #first item in each passed iterator is paired together, and then the second item
        #in each passed iterator are paired together etc.
        for (x, y, w, h), emotion in zip(faces, emotions):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text with the recognized emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    #ord returns ascii value of the character
    if key == ord("q"):
        break

#Releasing the camera
cam.release()
#Destroying all windows
cv2.destroyAllWindows()
