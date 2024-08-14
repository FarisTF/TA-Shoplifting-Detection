import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import time 
from playsound import playsound
import csv

tf.config.optimizer.set_jit(True)


colors = [(245,117,16), (16,34,245), (117,245,16)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

actions = np.array(['normal', 'shoplifting'])
res = [0,0]
threshold = 0.8

# Variabel toleransi flicker grabbing
countdown_grab = 0
is_masih_ditoleransi = False

# Text visibility
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50)
fontScale = 1
color = (255, 0, 0) 
thickness = 1

# Load the YOLOv8 model
pose = YOLO('model/yolov8n-pose.pt')
item = YOLO('model/4items_v1.pt')
model = tf.keras.models.load_model('LSTM_v11_10FPS.h5')
# item = YOLO('yolov8n.pt')

# Open the video file
# # video_path = "path/to/video.mp4"

# urlnya = "https://www.youtube.com/watch?v=AZ6toS8UZOE"

# url = pafy.new(urlnya).getbest(preftype='mp4').url


# cap = cv2.VideoCapture("rtsp://admin:password1@169.254.109.243/H.264")
# cap = cv2.VideoCapture("mushola_lab\shoplift_1_mushola_lab.avi")
# cap = cv2.VideoCapture("kamar_webcamlaptop\\shoplift_40.avi")
# cap = cv2.VideoCapture("C:/Users/faris/Downloads/mushola_gabung.mp4")
cap = cv2.VideoCapture(0)
# print(cap.getBackendName())
# cap = cv2.VideoCapture(5)

fps_arr = []

frame_num = 0
sequences = []
sentence = []

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0


result = cv2.VideoWriter(("testing_shoplift_kaka.avi"),  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            25, (960,540))

normal_warning = cv2.imread('klasifikasi_normal.png',1)
shoplifting_warning = cv2.imread('klasifikasi_shoplifting.png',1)
shoplift_frame_count = -1

# Loop through the video frames
while cap.isOpened():
    frame_num+=1
    # Read feed
    # Read a frame from the video
    success, frame = cap.read()
    action = None

    context_data = ["luar" for i in range(18)]

    if success:
        frame = cv2.resize(frame, (960,540))
        pose_results = pose(frame, max_det=1)
        pose_frame = pose_results[0].plot(boxes=False)

        context_data = [0 for i in range(18)]

        # result_keypoint = results[0].keypoints

        item_results = item.track(frame, max_det=6, conf=0.2)
        item_results[0].orig_img = pose_frame
        pose_item_frame = item_results[0].plot(font_size=10, pil=True)
        # convert dlu ke int soalnya ini lempar lemparan antara pillow dan cv2
        pose_item_frame = pose_item_frame.astype(np.uint8)

        list_of_bbox = item_results[0].boxes.xyxy

        # Logic cuman dijalanin klo ada orang
        if((pose_results[0].keypoints.conf)!=None):
            # context_data = ["dalem pose_results" for i in range(18)]

            koordinat_tangan_kiri  = [round(pose_results[0].keypoints.data[0][9][0].item()),  # Koordinat X wrist
                                    round(pose_results[0].keypoints.data[0][9][1].item())] # Koordinat Y wrist
            
            koordinat_tangan_kanan  = [round(pose_results[0].keypoints.data[0][10][0].item()),  # Koordinat X wrist
                                    round(pose_results[0].keypoints.data[0][10][1].item())] # Koordinat Y wrist
            
            is_kiri_grab = False
            is_kanan_grab = False

            context_data = [round(pose_results[0].keypoints.data[0][1][0].item()), round(pose_results[0].keypoints.data[0][1][1].item()),
                            round(pose_results[0].keypoints.data[0][5][0].item()), round(pose_results[0].keypoints.data[0][5][1].item()),
                            round(pose_results[0].keypoints.data[0][6][0].item()), round(pose_results[0].keypoints.data[0][6][1].item()),
                            round(pose_results[0].keypoints.data[0][7][0].item()), round(pose_results[0].keypoints.data[0][7][1].item()),
                            round(pose_results[0].keypoints.data[0][8][0].item()), round(pose_results[0].keypoints.data[0][8][1].item()),
                            round(pose_results[0].keypoints.data[0][9][0].item()), round(pose_results[0].keypoints.data[0][9][1].item()),
                            round(pose_results[0].keypoints.data[0][10][0].item()), round(pose_results[0].keypoints.data[0][10][1].item()),
                            0,0,
                            0,0]

            for bbox in list_of_bbox:
                # context_data = ["dalem list_of_bbox" for i in range(18)]

                lebar_bbox = bbox[2].item() - bbox[0].item()
                tinggi_bbox = bbox[3].item() - bbox[1].item()

                rata_rata_lt = (lebar_bbox+tinggi_bbox)/2

                # tambahan_lebar_bbox = round(0.5 * rata_rata_lt)
                # tambahan_tinggi_bbox = round(0.4 * rata_rata_lt)

                tambahan_lebar_bbox = 45
                tambahan_tinggi_bbox = 35


                # Menggambar area grabbing (lebih luas dari bbox item ori)
                cv2.rectangle(pose_item_frame, [round(bbox[0].item() - tambahan_lebar_bbox),
                                                round(bbox[1].item() - tambahan_tinggi_bbox)],
                                                [round(bbox[2].item() + tambahan_lebar_bbox),
                                                    round(bbox[3].item() + tambahan_tinggi_bbox)], 
                                                    (255, 255, 255) , thickness) 
            
                # Logic grabbing tangan kiri
                is_item_inside_left_hand_x_range = ((round(bbox[0].item() - tambahan_lebar_bbox)) <= koordinat_tangan_kiri[0]) and (koordinat_tangan_kiri[0] <= (round(bbox[2].item() + tambahan_lebar_bbox)))
                is_item_inside_left_hand_y_range = ((round(bbox[1].item() - tambahan_tinggi_bbox)) <= koordinat_tangan_kiri[1]) and (koordinat_tangan_kiri[1] <= (round(bbox[3].item() + tambahan_tinggi_bbox)))
                if(is_item_inside_left_hand_x_range and is_item_inside_left_hand_y_range):
                    is_kiri_grab = True
                    countdown_grab = 30
                    # image[130:330, 88:288] = cropped_left_hand_frame
                
                # Logic grabbing tangan kanan
                is_item_inside_right_hand_x_range = ((round(bbox[0].item() - tambahan_lebar_bbox)) <= koordinat_tangan_kanan[0]) and (koordinat_tangan_kanan[0] <= (round(bbox[2].item() + tambahan_lebar_bbox)))
                is_item_inside_right_hand_y_range = ((round(bbox[1].item() - tambahan_tinggi_bbox)) <= koordinat_tangan_kanan[1]) and (koordinat_tangan_kanan[1] <= (round(bbox[3].item() + tambahan_tinggi_bbox)))
                if(is_item_inside_right_hand_x_range and is_item_inside_right_hand_y_range):
                    is_kanan_grab = True
                    countdown_grab = 30
                    # image[130:330, 88:288] = cropped_left_hand_frame

                # Bikin strukdat context (lokasi joint yg relevan, lokasi grabbed product)
                """
                    [kp1, kp5, kp6, kp7, kp8, kp9, kp10, 
                    center_coord_grabbed_prod, 
                    center_coord_bag] for now kita 0-in dlu saja
                """
                is_masih_ditoleransi = countdown_grab > 0
                if (is_kanan_grab or is_kiri_grab):
                    context_data[14] = (round(bbox[2].item()-bbox[0].item()+bbox[0].item())) # X produk
                    context_data[15] = (round(bbox[3].item()-bbox[1].item()+bbox[1].item())) # Y produk
                    print(context_data)
                # context_data.append(0) # Untuk X bag
                # context_data.append(0) # Untuk Y bag
                # print(context_data)

            cv2.putText(pose_item_frame,( str(countdown_grab) + str(is_masih_ditoleransi) + " is_grab kiri: " + str(is_kiri_grab)+"   is_grab kanan: "+str(is_kanan_grab)), 
                        [20,50], font, fontScale, color, thickness, cv2.LINE_AA)
        
        image = pose_item_frame
        countdown_grab -= 1
        # utk simulasi ngelag klo realtime
        if(frame_num%1==0):
            sequences.append(context_data)
        sequences = sequences[-20:]

        if (len(sequences) == 20):
            # if biar tensorflownya gak dijalanin di setiap frame
            if(frame_num%2==0):
                sequence_holder = []
                for per_frame in sequences:
                    per_frame_holder = []
                    for i in range(len(per_frame)):
                        if(i in [0,2,4,6,8,10,12,14]):
                            if(per_frame[i] == 0):
                                per_frame_holder.append(float(-1))
                            else:
                                per_frame_holder.append(float((per_frame[i]/640)-0.5))
                        else:
                            if(per_frame[i] == 0):
                                per_frame_holder.append(float(-1))
                            else:
                                per_frame_holder.append((per_frame[i]/480)-0.5)
                    sequence_holder.append(per_frame_holder)
                # print(sequence_holder)

                res = model.predict(np.expand_dims(sequence_holder, axis=0))[0]
                print(actions[np.argmax(res)])
                print(res)
                # Perbandingan confidence level untuk kelas normal dan shoplifting
                if (res[0]<res[1]):
                    if(shoplift_frame_count<0):
                        playsound('alarm_shoplifting.mp3', block=False)
                    shoplift_frame_count = 30
                
            
            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        # # Pengisian action stack
        # if((pose_results[0].keypoints.conf)!=None):
        #     # tangan kiri
        #     visibility_tangan_kiri  = pose_results[0].keypoints.data[0][9][2]
        #     pose_item_frame = cv2.putText(pose_item_frame, str(visibility_tangan_kiri), org, font,  
        #            fontScale, color, thickness, cv2.LINE_AA) 
        #     # if (results[0].keypoints)

        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
    
        # converting the fps into integer 
        fps = int(fps) 

        mini_fps_arr = []
        mini_fps_arr.append(fps)

        fps_arr.append(mini_fps_arr)
    
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 
    
        # putting the FPS count on the frame 
        cv2.putText(image, fps, (500, 300), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        result.write(image) 
        # Display the annotated frame
        cv2.imshow("Main", image)

        # Untuk warning di sebelah
        if(shoplift_frame_count>0):
            cv2.imshow("Warning", shoplifting_warning)
            shoplift_frame_count -= 1
        else:
            cv2.imshow("Warning", normal_warning)
            shoplift_frame_count -= 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
    #     # Break the loop if the end of the video is reached
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # continue
        break
with open('eval_fps_final.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(['FPS'])
    write.writerows(fps_arr)

result.release() 
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()