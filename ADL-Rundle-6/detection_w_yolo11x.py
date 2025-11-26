import cv2
import numpy as np
import glob
import os
import onnxruntime as ort

# Configuration
IMG_DIRECTORY = 'img1'
MODEL_PATH = 'yolo11x.onnx'
OUTPUT_FILE = 'det/yolo11x/det.txt'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression
INPUT_SIZE = 640  # YOLO input size

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

def run_detection():
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Load ONNX model
    print(f"Loading model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # Get list of images
    images = sorted(glob.glob(os.path.join(IMG_DIRECTORY, '*.jpg')))
    if not images:
        print(f"Error: No images found in {IMG_DIRECTORY}")
        return
    
    print(f"Processing {len(images)} images...")
    
    all_detections = []
    
    for i, img_path in enumerate(images):
        frame_num = i + 1
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        blob, scale, pad_w, pad_h = preprocess_image(image, INPUT_SIZE)
        
        # Run inference
        outputs = session.run(None, {input_name: blob})
        
        # Postprocess (includes NMS)
        detections = postprocess_detections(outputs, scale, pad_w, pad_h, orig_w, orig_h, CONFIDENCE_THRESHOLD)
        
        # Format: frame obj_id bb_left bb_top bb_width bb_height confidence x y z
        for det in detections:
            x, y, w, h, conf = det
            # obj_id is -1 (placeholder, will be assigned by tracker), x, y, z are -1
            line = f"{frame_num} -1 {int(x)} {int(y)} {int(w)} {int(h)} {conf:.6f} -1 -1 -1"
            all_detections.append(line)
        
        if frame_num % 50 == 0:
            print(f"Processed frame {frame_num}/{len(images)}, detections: {len(detections)}")
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(all_detections))
    
    print(f"Detections saved to {OUTPUT_FILE}")
    print(f"Total detections: {len(all_detections)}")

run_detection()
