import numpy as np
import scipy.optimize
import sys
import os

sys.path.append('../2D_Kalman-Filter_TP1')
from KalmanFilter import Kalmanfilter

def load_detections(file_path):
    detections_by_frame = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            frame = int(parts[0])
            # Parse detection: [obj_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z]
            detection = np.array([
                int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]),
                float(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])
            ])
            if frame not in detections_by_frame:
                detections_by_frame[frame] = []
            detections_by_frame[frame].append(detection)
    
    for frame in detections_by_frame:
        detections_by_frame[frame] = np.array(detections_by_frame[frame])
    return detections_by_frame

def init_kalman_filter(x, y):
    k_filter = Kalmanfilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
    k_filter.xk[0], k_filter.xk[1] = x, y
    return k_filter

def get_center(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)

def initialize_first_frame(detections, k_filters):
    # detections: (N, 9) -> [id, x, y, w, h, ...]
    # Add frame column at index 0 -> [frame, id, x, y, w, h, ...]
    tracks = detections.copy()
    tracks = np.insert(tracks, 0, 1, axis=1) # Frame 1
    
    for i in range(len(tracks)):
        # Assign ID (1-based index for simplicity in this logic)
        tracks[i][1] = i + 1
        tracks[i][6] = 1 # Confidence flag
        
        # Initialize KF
        x, y, w, h = tracks[i][2], tracks[i][3], tracks[i][4], tracks[i][5]
        cx, cy = get_center(x, y, w, h)
        
        k_filter = init_kalman_filter(cx, cy)
        k_filter.predict() # Initial prediction
        
        # Update track with predicted state 
        pred_x = k_filter.xk[0][0]
        pred_y = k_filter.xk[1][0]
        
        tracks[i][2] = int(pred_x - w / 2)
        tracks[i][3] = int(pred_y - h / 2)
        
        k_filters[tracks[i][1]] = k_filter
        
    return tracks.astype(int)

def compute_iou(box1, box2):
    # box: [x, y, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    
    if union_area == 0: return 0
    return inter_area / union_area

def match_detections(tracks, detections):
    # tracks: [frame, id, x, y, w, h, ...] (10 cols)
    # detections: [id, x, y, w, h, ...] (9 cols)
    
    num_tracks = len(tracks)
    num_dets = len(detections)
    cost_matrix = np.zeros((num_tracks, num_dets))
    
    for t in range(num_tracks):
        for d in range(num_dets):
            # Track box is at indices 2,3,4,5 (shifted by 1 due to frame col)
            box_t = tracks[t][2:6] 
            # Detection box is at indices 1,2,3,4
            box_d = detections[d][1:5]
            
            iou = compute_iou(box_t, box_d)
            cost_matrix[t][d] = 1 - iou # Jaccard index
            
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def process_tracking_loop(detections_by_frame, output_file):
    # Initialize
    k_filters = {}
    if 1 not in detections_by_frame:
        print("Frame 1 not found.")
        return

    tracks = initialize_first_frame(detections_by_frame[1], k_filters)
    max_id = np.max(tracks[:, 1])
    
    # Save first frame
    with open(output_file, 'w') as f:
        np.savetxt(f, tracks, fmt='%d', delimiter=',')
    print(f"Frame 1 processed. Tracks: {len(tracks)}")

    # Loop through frames
    # We sort keys to ensure order, as frames are sequential but may have missing indices
    sorted_frames = sorted(detections_by_frame.keys())
    
    # Start from the second frame available
    for i in range(1, len(sorted_frames)):
        try:
            frame_num = sorted_frames[i]
            
            current_detections = detections_by_frame[frame_num]
            
            # Apply Kalman predict step for all existing tracks
            for track_id, kf in k_filters.items():
                kf.predict()
            
            row_ind, col_ind = match_detections(tracks, current_detections)
            matches = {c: r for r, c in zip(row_ind, col_ind)}
            
            next_tracks = current_detections.copy()
            next_tracks = np.insert(next_tracks, 0, frame_num, axis=1)
            
            for m in range(len(next_tracks)):
                next_tracks[m][6] = 1 # Conf flag
                
                # Get geometry from current detection
                det_x, det_y = next_tracks[m][2], next_tracks[m][3]
                det_w, det_h = next_tracks[m][4], next_tracks[m][5]
                det_center_x, det_center_y = get_center(det_x, det_y, det_w, det_h)
                
                if m in matches:
                    prev_track_idx = matches[m]
                    track_id = tracks[prev_track_idx][1]
                    next_tracks[m][1] = track_id # Inherit ID
                    
                    # Kalman Update for matched track
                    if track_id in k_filters:
                        kf = k_filters[track_id]
                        measurement = np.array([[det_center_x], [det_center_y]])
                        kf.update(measurement)
                        
                        # Update track geometry with smoothed position
                        est_cx = kf.xk[0][0]
                        est_cy = kf.xk[1][0]
                        
                        next_tracks[m][2] = int(est_cx - det_w / 2)
                        next_tracks[m][3] = int(est_cy - det_h / 2)
                else:
                    # Unmatched -> New Track
                    max_id += 1
                    next_tracks[m][1] = max_id
                    
                    # Initialize new KF
                    k_filters[max_id] = init_kalman_filter(det_center_x, det_center_y)
            
            # Update tracks variable
            tracks = next_tracks.astype(int)
            print(tracks)
            
            # Save
            with open(output_file, 'a') as f:
                np.savetxt(f, tracks, fmt='%d', delimiter=',')
        
        except KeyError:
            continue

# Run
file_path = 'det/Yolov5l/det.txt'
output_file = 'filtered_tracking_results.txt'

# Load detections from file
detections_by_frame = load_detections(file_path)

# Process tracking
process_tracking_loop(detections_by_frame, output_file)

