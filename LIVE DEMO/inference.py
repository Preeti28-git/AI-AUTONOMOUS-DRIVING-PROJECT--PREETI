import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

def process_lane_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Create a mask for the region of interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    
    # Define a more precise trapezoid ROI
    roi_vertices = np.array([[
        (width * 0.1, height),           # Bottom left
        (width * 0.45, height * 0.6),    # Top left
        (width * 0.55, height * 0.6),    # Top right
        (width * 0.9, height)            # Bottom right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform for line detection with adjusted parameters
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                           minLineLength=50, maxLineGap=100)
    
    return lines, height, width, roi_vertices

def draw_lanes(frame, lines, height, width, roi_vertices):
    if lines is None:
        return frame
    
    # Create a blank image for lanes
    lane_image = np.zeros_like(frame)
    
    # Draw ROI area with semi-transparent overlay
    roi_overlay = frame.copy()
    cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
    frame = cv2.addWeighted(frame, 0.9, roi_overlay, 0.1, 0)
    
    # Separate left and right lines
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter lines based on slope
        if slope < -0.3:  # Left lane
            left_lines.append(line[0])
        elif slope > 0.3:  # Right lane
            right_lines.append(line[0])
    
    # Draw left lane
    if left_lines:
        left_line = np.mean(left_lines, axis=0, dtype=np.int32)
        # Draw filled polygon for left lane
        pts_left = np.array([[left_line[0], left_line[1]], 
                           [left_line[2], left_line[3]], 
                           [left_line[2], height],
                           [left_line[0], height]], np.int32)
        cv2.fillPoly(lane_image, [pts_left], (0, 0, 255))
    
    # Draw right lane
    if right_lines:
        right_line = np.mean(right_lines, axis=0, dtype=np.int32)
        # Draw filled polygon for right lane
        pts_right = np.array([[right_line[0], right_line[1]], 
                            [right_line[2], right_line[3]], 
                            [right_line[2], height],
                            [right_line[0], height]], np.int32)
        cv2.fillPoly(lane_image, [pts_right], (255, 0, 0))
    
    # Add lane overlay with transparency
    result = cv2.addWeighted(frame, 1, lane_image, 0.3, 0)
    
    # Draw lane boundaries with solid lines
    if left_lines:
        cv2.line(result, (left_line[0], left_line[1]), 
                (left_line[2], left_line[3]), (0, 0, 255), 3)
    if right_lines:
        cv2.line(result, (right_line[0], right_line[1]), 
                (right_line[2], right_line[3]), (255, 0, 0), 3)
    
    return result

def calculate_steering_angle(lines, width):
    if lines is None or len(lines) < 2:
        return 0, "CENTER"
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.3:
            left_lines.append(line[0])
        elif slope > 0.3:
            right_lines.append(line[0])
    
    if not left_lines or not right_lines:
        return 0, "CENTER"
    
    # Calculate lane center
    left_center = np.mean(left_lines, axis=0)
    right_center = np.mean(right_lines, axis=0)
    lane_center = (left_center[0] + right_center[0]) / 2
    
    # Calculate steering angle
    image_center = width / 2
    offset = lane_center - image_center
    steering_angle = offset / (width / 2) * 45
    
    # Determine direction
    if abs(steering_angle) < 5:
        direction = "CENTER"
    elif steering_angle < 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    
    return steering_angle, direction

def draw_dashboard(frame, fps, steering_angle, direction, detections, safe_distance=True):
    height, width = frame.shape[:2]
    dashboard_height = 150
    dashboard = np.zeros((dashboard_height, width, 3), dtype=np.uint8)
    
    # Draw dashboard background
    cv2.rectangle(dashboard, (0, 0), (width, dashboard_height), (50, 50, 50), -1)
    
    # Draw FPS meter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(dashboard, fps_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw steering visualization
    center_x = width // 2
    wheel_y = dashboard_height // 2
    
    # Draw steering wheel
    cv2.circle(dashboard, (center_x, wheel_y), 40, (255, 255, 255), 2)
    
    # Draw steering direction indicator
    angle_rad = np.radians(steering_angle)
    end_x = int(center_x + 40 * np.sin(angle_rad))
    end_y = int(wheel_y - 40 * np.cos(angle_rad))
    cv2.line(dashboard, (center_x, wheel_y), (end_x, end_y), (0, 255, 0), 3)
    
    # Draw direction text with color coding
    direction_colors = {
        "LEFT": (0, 0, 255),
        "CENTER": (0, 255, 0),
        "RIGHT": (255, 0, 0)
    }
    direction_text = f"Steering: {direction} ({steering_angle:.1f}°)"
    cv2.putText(dashboard, direction_text, (center_x + 100, wheel_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, direction_colors[direction], 2)
    
    # Draw detection count
    det_text = f"Detections: {detections}"
    cv2.putText(dashboard, det_text, (width - 250, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Combine dashboard with frame
    result = np.vstack((frame, dashboard))
    return result

try:
    # Initialize video capture and model
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA is {'not ' if not cv2.cuda.getCudaEnabledDeviceCount() else ''}available")
    
    print("Loading YOLO model...")
    model = YOLO("m_model.pt")
    print("Model loaded successfully!")
    print(f"Model classes: {model.names}")

    video_path = "test_1.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f" - Resolution: {width}x{height}")
    print(f" - FPS: {fps}")
    print(f" - Total frames: {total_frames}")

    # Create window
    window_name = "Advanced Autonomous Driving System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 900)  # Adjusted for dashboard

    # Initialize colors for object detection
    COLORS = np.random.uniform(0, 255, size=(100, 3))

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached after {frame_count} frames")
            break

        try:
            # Process lane detection
            lines, height, width, roi = process_lane_detection(frame)
            
            # Calculate steering
            steering_angle, direction = calculate_steering_angle(lines, width)
            
            # Draw lanes
            frame_with_lanes = draw_lanes(frame, lines, height, width, roi)
            
            # Run YOLO inference
            results = model(frame, conf=0.25)
            
            # Process detections
            num_detections = 0
            for r in results:
                boxes = r.boxes
                num_detections = len(boxes)
                
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Generate color
                    color = tuple(map(int, COLORS[cls % len(COLORS)]))
                    
                    # Draw box and label
                    cv2.rectangle(frame_with_lanes, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    
                    # Add label with background
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        frame_with_lanes,
                        (x1, y1 - label_height - 5),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame_with_lanes,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
            
            # Calculate current FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            current_fps = frame_count / elapsed_time
            
            # Add dashboard
            final_frame = draw_dashboard(
                frame_with_lanes, 
                current_fps, 
                steering_angle, 
                direction, 
                num_detections
            )
            
            # Display the result
            cv2.imshow(window_name, final_frame)
            
            frame_count += 1
            
            # Print stats every second
            if current_time - last_print_time >= 1.0:
                print(f"Frame {frame_count}/{total_frames} - "
                      f"FPS: {current_fps:.2f} - "
                      f"Detections: {num_detections} - "
                      f"Steering: {direction} ({steering_angle:.1f}°)")
                last_print_time = current_time

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            continue

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    
    if 'frame_count' in locals() and 'start_time' in locals():
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\nProcessing completed:")
        print(f" - Total frames processed: {frame_count}")
        print(f" - Average FPS: {avg_fps:.2f}")
        print(f" - Total time: {total_time:.2f} seconds")