import cv2
import numpy as np
import os

# Nama folder tempat ngesavenya
DATA_PATH = os.path.join('lab_webcamlaptop') 

# Actions that we try to detect
actions = np.array(['shoplift', 'normal'])
# Jumlah klip utk per class
no_sequences = 15
# Panjang total sliding window utk masuk model sequence classification
sequence_length = 60

# Pembuatan folder
try: 
    os.makedirs(DATA_PATH)
except:
    pass

cap = cv2.VideoCapture(0)
# WIDTH = 1280
# HEIGHT = 720

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
size = (int(cap.get(3)) , int(cap.get(4)) )


# Loop through actions
for action in actions:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        clip_path = os.path.join(DATA_PATH, str(action) + "_" + (str(sequence) +"_" + (str(DATA_PATH)) + ".avi"))
        result = cv2.VideoWriter(clip_path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            30, size)
        
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):

            # Read feed
            ret, image = cap.read()

            # NEW Apply wait logic
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(3000)
            else: 
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            
            # # NEW Export keypoints
            # keypoints = extract_keypoints(results)
            # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            # np.save(npy_path, keypoints)
            result.write(image) 

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        result.release() 
                
cap.release()
cv2.destroyAllWindows()