import numpy as np
import scipy.optimize


file_path = 'det/Yolov5l/det.txt'
detections_by_frame = {}

# Load the detections from the file
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(' ')
        frame = int(parts[0])
        obj_id = int(parts[1])
        bb_left = int(parts[2])
        bb_top = int(parts[3])
        bb_width = int(parts[4])
        bb_height = int(parts[5])
        confidence = float(parts[6])
        x = int(parts[7])
        y = int(parts[8])
        z = int(parts[9])
        
        detection = np.array([obj_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z])
        
        if frame not in detections_by_frame:
            detections_by_frame[frame] = []
        detections_by_frame[frame].append(detection)

# Convert lists to numpy arrays
for frame in detections_by_frame:
    detections_by_frame[frame] = np.array(detections_by_frame[frame])

# Initialize tracks list with frame 1 object instances
tracks = detections_by_frame[1].copy()
for i in range(len(tracks)):
    tracks[i][0] = i + 1
    tracks[i][5] = 1 # " (conf) act as a flag 1"
tracks = np.insert(tracks, 0, 1, axis=1).astype(int)
print(tracks)

# Initialize output file and save first frame
output_file = 'tracking_results.txt'
with open(output_file, 'w') as f:
    np.savetxt(f, tracks, fmt='%d', delimiter=',')

# N.B: By testing similarity of frames 1 and 2 that have identic objects, we deduce the same objects aren't necessarily detected at the order across all frames
# (And for 2 frames with different object instances we tried 1 and 34)

for i in range(1, len(detections_by_frame) + 1): # Keys of the dict start with 1
    try: # Not all frame indices are present
        frame1 = detections_by_frame[i]
        frame2 = detections_by_frame[i+1]
        num_tracks = len(frame1)
        num_detections = len(frame2)

        # Calculate similarity
        similarity_matrix = np.zeros((num_tracks, num_detections))
        for n in range(num_tracks):
            for m in range(num_detections):
                box1 = frame1[n][1:5]
                box2 = frame2[m][1:5]

                # Conversion from height and width to bottom right corner
                box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
                box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

                # Calculate intersection coordinates
                x1_inter = max(box1[0], box2[0])
                y1_inter = max(box1[1], box2[1])
                x2_inter = min(box1[2], box2[2])
                y2_inter = min(box1[3], box2[3])
                
                # Calculate intersection area
                inter_width = max(0, x2_inter - x1_inter)
                inter_height = max(0, y2_inter - y1_inter)
                intersection_area = inter_width * inter_height
                
                # Calculate union area
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = box1_area + box2_area - intersection_area

                # Avoid division by zero
                if union_area == 0:
                    similarity_matrix[n][m] = 1
                
                similarity_matrix[n][m] = 1 - intersection_area / union_area

        # Apply hungarian algorithm
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(similarity_matrix) # We keep the default maximize=False because we want to minimize the jaccard index
        #if i == 1:
            #print(row_ind, col_ind)

        # Update tracks
        
        # Initialize next_tracks with the geometry/info from frame2
        next_tracks = frame2.copy()
        next_tracks = np.insert(next_tracks, 0, i + 1, axis=1)
        
        # Find the current maximum ID to generate new ones
        max_id = 0
        if len(tracks) > 0:
            max_id = np.max(tracks[:, 1])
            
        # Create a mapping from frame2 index (col) to frame1/tracks index (row)
        matches = {c: r for r, c in zip(row_ind, col_ind)}
        
        # Iterate through all detections in the new frame
        for m in range(len(next_tracks)):
            next_tracks[m][6] = 1 # " (conf) act as a flag 1"
            if m in matches:
                # If matched, inherit the ID from the previous track
                prev_track_idx = matches[m]
                next_tracks[m][1] = tracks[prev_track_idx][1]
            else:
                # If unmatched (new detection), assign a new ID
                max_id += 1
                next_tracks[m][1] = max_id
        
        # Update the tracks variable to the new state
        tracks = next_tracks.astype(int)
        print(tracks)

        # Append current frame tracks to file
        with open(output_file, 'a') as f:
            np.savetxt(f, tracks, fmt='%d', delimiter=',')
    
    except KeyError:
        continue

