import cv2
from ultralytics import YOLO

# Text visibility
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 2

# Array of person, per person ada queue untuk tangan kanan dan kiri -> nanti
# Karna sekarang cuman 1 orang, bikin kiri dan kanan terpisah
action_queue_left = []
action_queue_right = []

# Load the YOLOv8 model
pose = YOLO('yolov8n-pose.pt')
item = YOLO('yolov8n.pt')

# Open the video file
# video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    action = None

    if success:
        pose_results = pose(frame, max_det=1)
        pose_frame = pose_results[0].plot(boxes=False)

        # result_keypoint = results[0].keypoints
        # print(results)

        item_results = item(pose_frame, max_det=2)
        # print(item_results[0].boxes)
        pose_item_frame = item_results[0].plot()

        # Pengisian action stack
        if((pose_results[0].keypoints.conf)!=None):
            # tangan kiri
            visibility_tangan_kiri  = pose_results[0].keypoints.data[0][9][2]
            print(visibility_tangan_kiri)
            pose_item_frame = cv2.putText(pose_item_frame, str(visibility_tangan_kiri), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
            # if (results[0].keypoints)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Pose", pose_item_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()