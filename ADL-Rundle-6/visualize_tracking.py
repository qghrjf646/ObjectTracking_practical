import cv2
import numpy as np
import os
import glob

IMG_DIRECTORY = 'img1'

# IoU-based tracker results
TRACK_FILE = 'tracking_results.txt'
OUTPUT_VIDEO = 'tracking_output.mp4'

# IoU + Kalman Filter tracker results
FILTERED_TRACK_FILE = 'filtered_tracking_results.txt'
FILTERED_OUTPUT_VIDEO = 'filtered_tracking_output.mp4'

# Appearance-aware tracker results (Yolov5l detections)
APPEARANCE_TRACK_FILE = 'appearance_tracking_results.txt'
APPEARANCE_OUTPUT_VIDEO = 'appearance_tracking_output.mp4'

# End-to-end (YOLO11x detection + appearance-aware tracking)
END_TO_END_TRACK_FILE = 'end_to_end_results.txt'
END_TO_END_OUTPUT_VIDEO = 'end_to_end_output.mp4'

# Appearance-aware tracker with YOLO11x detections
APPEARANCE_YOLO11X_TRACK_FILE = 'appearance_yolo11x_results.txt'
APPEARANCE_YOLO11X_OUTPUT_VIDEO = 'appearance_yolo11x_output.mp4'

# Mapping from file to window name
FILE_TO_WINDOW = {
    TRACK_FILE: "IoU Tracking Results",
    FILTERED_TRACK_FILE: "IoU + KF Tracking Results",
    APPEARANCE_TRACK_FILE: "Appearance-Aware Tracking (Yolov5l)",
    END_TO_END_TRACK_FILE: "End-to-End Tracking (YOLO11x)",
    APPEARANCE_YOLO11X_TRACK_FILE: "Appearance-Aware Tracking (YOLO11x)",
}

def visualize_tracking(results_file, img_dir, output_filename):
    # 1. Load Tracking Results
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    # Load data: frame, id, left, top, width, height, ...
    data = np.loadtxt(results_file, delimiter=',')
    
    # Organize tracks by frame for easy access
    tracks_by_frame = {}
    for row in data:
        frame_idx = int(row[0])
        if frame_idx not in tracks_by_frame:
            tracks_by_frame[frame_idx] = []
        tracks_by_frame[frame_idx].append(row)

    # 2. Setup Video Writer
    # Get list of images to determine video properties
    # Assuming images are named like 000001.jpg, 000002.jpg
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    
    if not images:
        print(f"Error: No images found in {img_dir}")
        return

    # Read first image to get dimensions
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape
    size = (width, height)
    
    # Initialize VideoWriter (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, 30, size)

    # Determine window name based on input file
    window_name = FILE_TO_WINDOW.get(results_file, "Tracking Results")

    # Create named windows and resize them (e.g., to 75% of the original resolution)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(width * 0.75), int(height * 0.75))

    # Generate random colors for IDs
    np.random.seed(42)
    colors = np.random.randint(0, 255, (int(data[:, 1].max()) + 100, 3))

    print(f"Processing {len(images)} frames...")

    # 3. Process Frames
    for i, img_path in enumerate(images):
        frame_num = i + 1
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        # Draw tracks if they exist for this frame
        if frame_num in tracks_by_frame:
            for track in tracks_by_frame[frame_num]:
                track_id = int(track[1])
                x = int(track[2])
                y = int(track[3])
                w = int(track[4])
                h = int(track[5])

                color = [int(c) for c in colors[track_id % len(colors)]]
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw Track ID background and text
                label = f"ID: {track_id}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x, y - 20), (x + text_w, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add frame number info
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 4. Save and Display
        out.write(frame)
        
        cv2.imshow(window_name, frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    out.release()
    print(f"Video saved to {output_filename}")
    
    # Wait for a key press before closing the window
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Step 2: IoU-based tracker
#visualize_tracking(TRACK_FILE, IMG_DIRECTORY, OUTPUT_VIDEO)

# Step 3:  IoU + Kalman Filter tracker
#visualize_tracking(FILTERED_TRACK_FILE, IMG_DIRECTORY, FILTERED_OUTPUT_VIDEO)

# Step 4: Appearance-aware tracker (Yolov5l detections)
#visualize_tracking(APPEARANCE_TRACK_FILE, IMG_DIRECTORY, APPEARANCE_OUTPUT_VIDEO)

# Bonus End-to-end (YOLO11x detection + appearance-aware tracking)
visualize_tracking(END_TO_END_TRACK_FILE, IMG_DIRECTORY, END_TO_END_OUTPUT_VIDEO)
#visualize_tracking(APPEARANCE_YOLO11X_TRACK_FILE, IMG_DIRECTORY, APPEARANCE_YOLO11X_OUTPUT_VIDEO)