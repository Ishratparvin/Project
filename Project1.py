from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    result = er.recognise_emotion(frame,return_type='GBR')
    
    if result is not None and len(result) == 2:
        faces, emotions = result

        for (x, y, w, h), emotion in zip(faces, emotions):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text with the recognized emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
