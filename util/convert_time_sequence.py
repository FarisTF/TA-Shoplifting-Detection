import cv2
from ultralytics import YOLO
import numpy 
import os

"""
Struktur folder yang dimasukkin harus
    -Parent
        -Shoplift
            -Clip1
            -Clip2
            -Clip3...
        -Normal
            -Clip1
            -Clip2
            -Clip3...
"""

nama_folder = "FINAL_clean binary"


# Text visibility
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50)
fontScale = 0.7
color = (255, 0, 0) 
thickness = 1

# Load the YOLOv8 model
pose = YOLO('model/yolov8n-pose.pt')
item = YOLO('model/4items_v1.pt')

def process_clip(nama_clip):
    sequences = []
    cap = cv2.VideoCapture(nama_clip)
    size = (int(cap.get(3)) , int(cap.get(4)) )
    result = cv2.VideoWriter((nama_clip+"_anotated.avi"),  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                25, size)
    # Variabel toleransi flicker grabbing
    countdown_grab = 0
    is_masih_ditoleransi = False

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        action = None

        context_data = ["luar" for i in range(18)]

        if success:
            pose_results = pose(frame, max_det=1)
            pose_frame = pose_results[0].plot(boxes=False)

            context_data = [0 for i in range(18)]

            # result_keypoint = results[0].keypoints

            item_results = item.track(frame, max_det=4, conf=0.05)
            item_results[0].orig_img = pose_frame
            pose_item_frame = item_results[0].plot(font_size=10, pil=True)
            # convert dlu ke int soalnya ini lempar lemparan antara pillow dan cv2
            pose_item_frame = pose_item_frame.astype(numpy.uint8)

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
                    is_kiri_grab = False
                    is_kanan_grab = False
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
                    

                cv2.putText(pose_item_frame,( str(countdown_grab) + str(is_masih_ditoleransi) + " is_grab kiri: " + str(is_kiri_grab)+"   is_grab kanan: "+str(is_kanan_grab)), 
                            [20,50], font, fontScale, color, thickness, cv2.LINE_AA)
            sequences.append(context_data)
            countdown_grab -= 1

            # # Pengisian action stack
            # if((pose_results[0].keypoints.conf)!=None):
            #     # tangan kiri
            #     visibility_tangan_kiri  = pose_results[0].keypoints.data[0][9][2]
            #     pose_item_frame = cv2.putText(pose_item_frame, str(visibility_tangan_kiri), org, font,  
            #            fontScale, color, thickness, cv2.LINE_AA) 
            #     # if (results[0].keypoints)

            # Display the annotated frame
            cv2.imshow(nama_clip, pose_item_frame)
            result.write(pose_item_frame) 

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
        #     # Break the loop if the end of the video is reached
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return(sequences)

try: 
    os.makedirs(os.path.join(nama_folder, "context_data"))
except:
    pass
for subdir, dirs, files in os.walk(nama_folder):
    for file in files:
        sequences = process_clip(os.path.join(subdir, file))
        sequences_numpy = numpy.array(sequences)
        numpy.save(os.path.join(subdir,"context_data\\", file), sequences_numpy)
    print("\n\n")




