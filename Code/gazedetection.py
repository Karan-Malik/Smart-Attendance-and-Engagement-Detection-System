import cv2
from gaze_tracking import GazeTracking


def recognize_gaze():
    gaze = GazeTracking()
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640) 
    cam.set(4, 480) 
    
    while True:
        _, frame = cam.read()
        gaze.refresh(frame)
    
        new_frame = gaze.annotated_frame()
        text = ""
    
        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
    
        cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Demo", new_frame)
    
        if (cv2.waitKey(1) == ord('q')):
            break