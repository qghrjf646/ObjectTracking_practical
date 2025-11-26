import numpy as np
import scipy.optimize
import sys
import cv2
import glob
import os
import onnxruntime as ort

ALPHA = 0.5  # Weight for IoU
BETA = 0.5   # Weight for appearance

# Configuration
IMG_DIRECTORY = 'img1'
YOLO_MODEL_PATH = 'yolo11x.onnx'
REID_MODEL_PATH = '../reid_osnet_x025_market1501.onnx'
OUTPUT_FILE = 'end_to_end_results.txt'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression
INPUT_SIZE = 640  # YOLO input size

# Ensure the path to KalmanFilter is correct
sys.path.append('../2D_Kalman-Filter_TP1')
from KalmanFilter import Kalmanfilter


def preprocess_image(image, input_size):
    h, w = image.shape[:2]
    
    # Calculate scale to fit input size while maintaining aspect ratio
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image (letterbox)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    # Convert to float and normalize
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
    blob = np.expand_dims(blob, axis=0)   # Add batch dimension
    
    return blob, scale, pad_w, pad_h

def apply_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    
    # Convert to format expected by cv2.dnn.NMSBoxes: [x, y, w, h]
    boxes_for_nms = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    scores_list = [float(s) for s in scores]
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, CONFIDENCE_THRESHOLD, iou_threshold)
    
    # Handle different OpenCV versions (some return nested list, some return flat array)
    if len(indices) > 0:
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]
        else:
            indices = list(indices)
    
    return indices

def postprocess_detections(outputs, scale, pad_w, pad_h, orig_w, orig_h, conf_threshold):
    # YOLO11 output shape: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = outputs[0].squeeze().T
    
    boxes = []
    scores = []
    
    for pred in predictions:
        # First 4 values are box coordinates (cx, cy, w, h)
        cx, cy, w, h = pred[:4]
        # Remaining 80 values are class scores
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        # Filter by confidence and only keep person class (class 0 in COCO)
        if confidence >= conf_threshold and class_id == 0:
            # Convert from center format to top-left format
            # Also adjust for padding and scale
            x1 = (cx - w / 2 - pad_w) / scale
            y1 = (cy - h / 2 - pad_h) / scale
            box_w = w / scale
            box_h = h / scale
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            box_w = min(box_w, orig_w - x1)
            box_h = min(box_h, orig_h - y1)
            
            boxes.append([x1, y1, box_w, box_h])
            scores.append(confidence)
    
    # Apply Non-Maximum Suppression
    nms_indices = apply_nms(boxes, scores, NMS_THRESHOLD)
    
    # Return only the boxes that survived NMS
    detections = []
    for idx in nms_indices:
        x, y, w, h = boxes[idx]
        conf = scores[idx]
        detections.append([x, y, w, h, conf])
    
    return detections

def run_detection_on_frame(yolo_session, yolo_input_name, image):
    orig_h, orig_w = image.shape[:2]
    
    # Preprocess
    blob, scale, pad_w, pad_h = preprocess_image(image, INPUT_SIZE)
    
    # Run inference
    outputs = yolo_session.run(None, {yolo_input_name: blob})
    
    # Postprocess
    detections = postprocess_detections(outputs, scale, pad_w, pad_h, orig_w, orig_h, CONFIDENCE_THRESHOLD)
    
    # Convert to array format: [obj_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z]
    det_array = []
    for j, det in enumerate(detections):
        x, y, w, h, conf = det
        det_array.append([j + 1, int(x), int(y), int(w), int(h), conf, -1, -1, -1])
    
    return np.array(det_array) if det_array else None


def init_kalman_filter(x, y):
    k_filter = Kalmanfilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
    k_filter.xk[0], k_filter.xk[1] = x, y
    return k_filter

def get_center(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)

def initialize_first_frame(detections, k_filters, frame_num=1):
    # detections: (N, 9) -> [id, x, y, w, h, ...]
    # Add frame column at index 0 -> [frame, id, x, y, w, h, ...]
    tracks = detections.copy()
    tracks = np.insert(tracks, 0, frame_num, axis=1)
    
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
    
    tracks = tracks.astype(int)
    print(tracks)
    return tracks

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
        im_crops = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
    else:
        im_crops = frame[y1:y2, x1:x2]
        
    # Double check size
    if im_crops.size == 0:
        im_crops = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
    
    roi_input = cv2.resize(im_crops, (roi_width, roi_height)) 
    roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB) 
    roi_input = (np.asarray(roi_input).astype(np.float32) - roi_means) / roi_stds 
    roi_input = np.moveaxis(roi_input, -1, 0) 
    object_patch = roi_input.astype('float32') 
    return object_patch

def match_detections(tracks, detections, frame, reid_session, reid_input_name):
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

            # Process patch of each box
            processed_patch_t = preprocess_patch(64, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], box_t, frame)
            processed_patch_d = preprocess_patch(64, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], box_d, frame)
            
            # Get features from ReID model
            input_t = np.expand_dims(processed_patch_t, axis=0)
            input_d = np.expand_dims(processed_patch_d, axis=0)
            features_t = reid_session.run(None, {reid_input_name: input_t})[0]
            features_d = reid_session.run(None, {reid_input_name: input_d})[0]
            
            cosine_sim = np.dot(features_t, features_d.T) / (np.linalg.norm(features_t) * np.linalg.norm(features_d) + 1e-6)
            norm_similarity = cosine_sim
            
            score = ALPHA * iou + BETA * norm_similarity
            cost_matrix[t][d] = score

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
    return row_ind, col_ind


def run_end_to_end():
    # Load models
    print("Loading YOLO model...")
    yolo_session = ort.InferenceSession(YOLO_MODEL_PATH)
    yolo_input_name = yolo_session.get_inputs()[0].name
    
    print("Loading ReID model...")
    reid_session = ort.InferenceSession(REID_MODEL_PATH)
    reid_input_name = reid_session.get_inputs()[0].name
    
    # Get list of images
    images = sorted(glob.glob(os.path.join(IMG_DIRECTORY, '*.jpg')))
    if not images:
        print(f"Error: No images found in {IMG_DIRECTORY}")
        return
    
    print(f"Processing {len(images)} images...")
    
    k_filters = {}
    tracks = None
    max_id = 0
    first_frame_processed = False
    
    for i, img_path in enumerate(images):
        frame_num = i + 1
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Run detection
        detections = run_detection_on_frame(yolo_session, yolo_input_name, frame)
        
        if detections is None or len(detections) == 0:
            print(f"Frame {frame_num}: No detections")
            continue
        
        if not first_frame_processed:
            # Initialize first frame
            tracks = initialize_first_frame(detections, k_filters, frame_num)
            max_id = np.max(tracks[:, 1])
            
            # Save first frame
            with open(OUTPUT_FILE, 'w') as f:
                np.savetxt(f, tracks, fmt='%d', delimiter=',')
            print(f"Frame {frame_num} processed. Tracks: {len(tracks)}")
            first_frame_processed = True
            continue
        
        # Predict for all existing tracks before matching
        for track_id, kf in k_filters.items():
            kf.predict()
        
        # Match detections
        row_ind, col_ind = match_detections(tracks, detections, frame, reid_session, reid_input_name)
        matches = {c: r for r, c in zip(row_ind, col_ind)}
        
        # Prepare next_tracks array
        next_tracks = detections.copy()
        next_tracks = np.insert(next_tracks, 0, frame_num, axis=1)
        
        for m in range(len(next_tracks)):
            next_tracks[m][6] = 1 # Conf flag
            
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
        
        with open(OUTPUT_FILE, 'a') as f:
            np.savetxt(f, tracks, fmt='%d', delimiter=',')
    
    print(f"End-to-end tracking complete. Results saved to {OUTPUT_FILE}")

run_end_to_end()
