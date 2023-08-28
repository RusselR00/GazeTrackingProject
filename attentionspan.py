import cv2
from gaze_tracking import GazeTracking
from module_name import Calibration
import time

# Constants for split-screen layout
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SPLIT_SCREEN_WIDTH = int(FRAME_WIDTH / 2)

# Initialize gaze tracking and calibration
gaze = GazeTracking()
calibration = Calibration()

webcam = cv2.VideoCapture(0)

# Variables for attention span calculation
start_time_left = None
start_time_right = None
attention_span_left = 0
attention_span_right = 0

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    # Perform calibration if not complete
    if not calibration.is_complete():
        eye_frame = gaze.annotated_frame()
        side = gaze.eye_side()
        calibration.evaluate(eye_frame, side)

        if calibration.is_complete():
            print("Calibration is complete. Starting gaze tracking...")
            gaze.set_thresholds(calibration.threshold(0), calibration.threshold(1))

    # Analyze frame with gaze tracking
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking left"
        if start_time_left is None:
            start_time_left = time.time()
        start_time_right = None
    elif gaze.is_left():
        text = "Looking right"
        if start_time_right is None:
            start_time_right = time.time()
        start_time_left = None
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display split-screen videos
    left_video = frame[:, :SPLIT_SCREEN_WIDTH, :]
    right_video = frame[:, SPLIT_SCREEN_WIDTH:, :]
    cv2.imshow("Left Video", left_video)
    cv2.imshow("Right Video", right_video)

    # Calculate attention span
    if start_time_left is not None:
        attention_span_left += time.time() - start_time_left
    if start_time_right is not None:
        attention_span_right += time.time() - start_time_right

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

# Print attention span results
print("Attention span for left video:", attention_span_left, "seconds")
print("Attention span for right video:", attention_span_right, "seconds")
