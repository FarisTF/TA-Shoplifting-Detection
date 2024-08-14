# import the opencv library 
import cv2 
  
# define a video capture object 
cap = cv2.VideoCapture(1) 
size = (int(cap.get(3)) , int(cap.get(4)) )
result = cv2.VideoWriter("test.avi",  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            30, size) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = cap.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    result.write(frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

result.release()
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 