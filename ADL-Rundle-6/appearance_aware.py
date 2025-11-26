import numpy as np
import scipy.optimize
import sys
import cv2
import onnxruntime as ort
import os

ALPHA = 0.5  # Weight for IoU
BETA = 0.5   # Weight for appearance

sys.path.append('../2D_Kalman-Filter_TP1')
from KalmanFilter import Kalmanfilter

# Exact same logic as IoU_KF_tracker.py but with matching taking appearance in consideration
def load_detections(file_path):
    detections_by_frame = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            frame = int(parts[0])
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
    tracks = detections.copy()
    tracks = np.insert(tracks, 0, 1, axis=1) 
    for i in range(len(tracks)):
        tracks[i][1] = i + 1
        tracks[i][6] = 1 
        x, y, w, h = tracks[i][2], tracks[i][3], tracks[i][4], tracks[i][5]
        cx, cy = get_center(x, y, w, h)
        k_filter = init_kalman_filter(cx, cy)
        k_filter.predict() 
        pred_x = k_filter.xk[0][0]
        pred_y = k_filter.xk[1][0]
        tracks[i][2] = int(pred_x - w / 2)
        tracks[i][3] = int(pred_y - h / 2)
        k_filters[tracks[i][1]] = k_filter
    tracks = tracks.astype(int)
    print(tracks)
    return tracks

def compute_iou(box1, box2):
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

def preprocess_patch(roi_width, roi_height, roi_means, roi_stds, bounding_box, frame):
        img_h, img_w, _ = frame.shape
        x, y, w, h = bounding_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Clamp coordinates to be within image dimensions
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        # Check if the resulting crop is valid (non-empty)
        if x2 <= x1 or y2 <= y1:
            # Return a black patch if the box is completely outside or invalid
            im_crops = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
        else:
            im_crops = frame[y1:y2, x1:x2]
            
        # Double check size (e.g. if w or h were 0 originally)
        if im_crops.size == 0:
             im_crops = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)

        # Apply given transformations to the cropped image
        roi_input = cv2.resize(im_crops, (roi_width, roi_height)) 
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB) 
        roi_input = (np.asarray(roi_input).astype(np.float32) - roi_means) / roi_stds 
        roi_input = np.moveaxis(roi_input, -1, 0) 
        object_patch = roi_input.astype('float32') 
        return object_patch

def match_detections(tracks, detections):
    # tracks: [frame, id, x, y, w, h, ...] (10 cols)
    # detections: [id, x, y, w, h, ...] (9 cols)
    
    num_tracks = len(tracks)
    num_dets = len(detections)
    cost_matrix = np.zeros((num_tracks, num_dets))
    
    # Get the input name expected by the ONNX model dynamically
    input_name = reid_session.get_inputs()[0].name
    
    for t in range(num_tracks):
        for d in range(num_dets):
            # Track box is at indices 2,3,4,5 (shifted by 1 due to frame col)
            box_t = tracks[t][2:6] 
            # Detection box is at indices 1,2,3,4
            box_d = detections[d][1:5]
            
            iou = compute_iou(box_t, box_d)
            
            # Get frame by loading the image file
            img_path = f'img1/{tracks[t][0]:06d}.jpg'
            frame = cv2.imread(img_path)

            # Process patch of each box
            processed_patch_t = preprocess_patch(64, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], box_t, frame) # correct params
            processed_patch_d = preprocess_patch(64, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], box_d, frame)
            
            # Get features from ReID model
            input_t = np.expand_dims(processed_patch_t, axis=0)
            input_d = np.expand_dims(processed_patch_d, axis=0)
            features_t = reid_session.run(None, {input_name: input_t})[0]
            features_d = reid_session.run(None, {input_name: input_d})[0]
            # euclid_dist = np.linalg.norm(features_t - features_d)
            cosine_sim = np.dot(features_t, features_d.T) / (np.linalg.norm(features_t) * np.linalg.norm(features_d) + 1e-6)
            #norm_similarity = 1 / (1 + euclid_dist)
            norm_similarity = cosine_sim
            score = ALPHA * iou + BETA * norm_similarity
            cost_matrix[t][d] = score

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True) # We want to maximize the overall similarity
    return row_ind, col_ind

def process_tracking_loop(detections_by_frame, output_file):
    k_filters = {}
    if 1 not in detections_by_frame:
        print("Frame 1 not found.")
        return
    tracks = initialize_first_frame(detections_by_frame[1], k_filters)
    max_id = np.max(tracks[:, 1])
    with open(output_file, 'w') as f:
        np.savetxt(f, tracks, fmt='%d', delimiter=',')
    print(f"Frame 1 processed. Tracks: {len(tracks)}")
    sorted_frames = sorted(detections_by_frame.keys())
    for i in range(1, len(sorted_frames)):
        try: 
            frame_num = sorted_frames[i]
            current_detections = detections_by_frame[frame_num]
            for track_id, kf in k_filters.items():
                kf.predict()
            row_ind, col_ind = match_detections(tracks, current_detections)
            matches = {c: r for r, c in zip(row_ind, col_ind)}
            next_tracks = current_detections.copy()
            next_tracks = np.insert(next_tracks, 0, frame_num, axis=1)
            for m in range(len(next_tracks)):
                next_tracks[m][6] = 1 
                det_x, det_y = next_tracks[m][2], next_tracks[m][3]
                det_w, det_h = next_tracks[m][4], next_tracks[m][5]
                det_center_x, det_center_y = get_center(det_x, det_y, det_w, det_h)
                if m in matches:
                    prev_track_idx = matches[m]
                    track_id = tracks[prev_track_idx][1]
                    next_tracks[m][1] = track_id 
                    if track_id in k_filters:
                        kf = k_filters[track_id]
                        measurement = np.array([[det_center_x], [det_center_y]])
                        kf.update(measurement)
                        est_cx = kf.xk[0][0]
                        est_cy = kf.xk[1][0]
                        next_tracks[m][2] = int(est_cx - det_w / 2)
                        next_tracks[m][3] = int(est_cy - det_h / 2)
                else:
                    max_id += 1
                    next_tracks[m][1] = max_id
                    k_filters[max_id] = init_kalman_filter(det_center_x, det_center_y)
            tracks = next_tracks.astype(int)
            print(tracks)
            with open(output_file, 'a') as f:
                np.savetxt(f, tracks, fmt='%d', delimiter=',')
        except KeyError:
            continue

# Configuration
REID_MODEL_PATH = '../reid_osnet_x025_market1501.onnx'

# Load ReID model in onnx format (global for use in match_detections)
reid_session = ort.InferenceSession(REID_MODEL_PATH)

def run_tracking(detection_file, output_file):
    print(f"Running tracking on {detection_file}")
    detections_by_frame = load_detections(detection_file)
    process_tracking_loop(detections_by_frame, output_file)
    print(f"Results saved to {output_file}")

# run_tracking('det/Yolov5l/det.txt', 'appearance_tracking_results.txt')
    
# Run with YOLO11x detections (if available)
yolo11x_det_file = 'det/yolo11x/det.txt'
run_tracking(yolo11x_det_file, 'appearance_yolo11x_results.txt')

