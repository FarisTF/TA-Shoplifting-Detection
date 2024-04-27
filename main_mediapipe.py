import cv2
import mediapipe as mp
from ultralytics import YOLO

# Text visibility
font = cv2.FONT_HERSHEY_SIMPLEX 
org = [50, 50]
fontScale = 1
color = (255, 0, 0) 
thickness = 2

count = 0

# Array of person, per person ada queue untuk tangan kanan dan kiri -> nanti
# Karna sekarang cuman 1 orang, bikin kiri dan kanan terpisah
action_queue_left = []
action_queue_right = []

item = YOLO('yolov8n.pt')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,model_complexity=0)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    image = cv2.imread(r'data window.png')
    # Read frame from webcam
    ret, frame = cap.read()

    if ret:
        # print(frame.shape)

        item_results = item.track(frame, max_det=1) #39 itu index class bottle
        item_frame = item_results[0].plot()

        # Convert frame to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        item_frame = cv2.cvtColor(item_frame, cv2.COLOR_BGR2RGB)

        # Detect pose landmarks from the input image
        detection_result = detector.process(image_rgb)

        # Draw landmarks on the image
        if detection_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(item_frame, detection_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Pengisian action stack
            # tangan kiri
            is_grab = False
            koordinat_tangan_kiri  = [round(detection_result.pose_landmarks.landmark[19].x * frame.shape[1]), 
                                    round(detection_result.pose_landmarks.landmark[19].y * frame.shape[0])]

            # print((koordinat_tangan_kiri))

            list_of_bbox = item_results[0].boxes.xyxy
            print(item_results is None)

            for bbox in list_of_bbox:
                is_item_inside_hand_x_range = (bbox[0] <= koordinat_tangan_kiri[0]) and (koordinat_tangan_kiri[0] <= bbox[2])
                is_item_inside_hand_y_range = (bbox[1] <= koordinat_tangan_kiri[1]) and (koordinat_tangan_kiri[1] <= bbox[3])
                if(is_item_inside_hand_x_range and is_item_inside_hand_y_range):
                    is_grab = True
                    
                    cropped_left_hand_frame = frame[koordinat_tangan_kiri[1]-100 : koordinat_tangan_kiri[1]+100, 
                                            koordinat_tangan_kiri[0]-100 : koordinat_tangan_kiri[0]+100]
                    
                    # image[130:330, 88:288] = cropped_left_hand_frame
                    


            

            # for i in range 

            count+=1
            
            image_rgb = cv2.putText(item_frame, str(is_grab), org, font,  
                fontScale, color, thickness, cv2.LINE_AA)
            image_rgb = cv2.putText(item_frame, str(koordinat_tangan_kiri), koordinat_tangan_kiri, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
            # if (results[0].keypoints)

        # Display the annotated image
        cv2.imshow('Shoplifting Detection', cv2.cvtColor(item_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('Data', image)

    # Break gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()

cv2.destroyAllWindows()
