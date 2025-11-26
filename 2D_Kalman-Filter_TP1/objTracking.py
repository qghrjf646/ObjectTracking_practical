import numpy as np
import cv2

from KalmanFilter import Kalmanfilter
from Detector import detect

# Initialize Kalman Filter
k_filter = Kalmanfilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

video_path = "video/randomball.avi"
cap = cv2.VideoCapture(video_path)

trajectory = []

while True:
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
        
    centers = detect(frame)
    if len(centers) != 0:
        k_filter.predict()
        predicted = k_filter.xk.copy()

        # Use the first detected center (convert to proper shape)
        measurement = centers[0]  # This should be np.array([[x], [y]])
        k_filter.update(measurement)
        estimated = k_filter.xk.copy()
        
        # Extract coordinates
        detected_x, detected_y = int(measurement[0][0]), int(measurement[1][0])
        predicted_x, predicted_y = int(predicted[0][0]), int(predicted[1][0])
        estimated_x, estimated_y = int(estimated[0][0]), int(estimated[1][0])
        
        # Draw detected circle (green color)
        cv2.circle(frame, (detected_x, detected_y), 10, (0, 255, 0), 2)
        
        # Draw blue rectangle as predicted object position
        cv2.rectangle(frame, (predicted_x-15, predicted_y-15), 
                      (predicted_x+15, predicted_y+15), (255, 0, 0), 2)
        
        # Draw red rectangle as estimated object position
        cv2.rectangle(frame, (estimated_x-15, estimated_y-15), 
                      (estimated_x+15, estimated_y+15), (0, 0, 255), 2)
        
        # Draw trajectory (tracking path)
        trajectory.append((estimated_x, estimated_y))
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 2)
        
        # So we can add text labels, but it makes the output cluttered
        # cv2.putText(frame, "Detected", (detected_x+15, detected_y-15), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.putText(frame, "Predicted", (predicted_x+15, predicted_y-15), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(frame, "Estimated", (estimated_x+15, estimated_y-15), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    cv2.imshow('Kalman Filter Tracking', frame)
    
    # Exit conditions
    if cv2.waitKey(25) & 0xFF == ord('q'):
     break
    

# Release everything when done
cap.release()

# cv2.destroyAllWindows() # So we can observe the final result