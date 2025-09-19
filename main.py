import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
import cv2

# --- Configuration & Initialization ---
app = FastAPI()
mp_pose = mp.solutions.pose
SMOOTHING_FACTOR = 0.05
UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"

# --- Configuration for Reel/TikTok Detection ---
BLUR_THRESHOLD = 100 
SIDEBAR_WIDTH_PERCENT = 0.25 

# --- NEW: Configuration for Split-Screen Detection ---
# A peak in edges must be this many times stronger than the average
# to be considered a split-screen boundary. 10 is a good starting point.
SPLIT_PEAK_FACTOR = 8.0

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --- Helper functions for blur detection ---
def _calculate_blurriness(frame_section: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_section, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_vertical_reel(frame: np.ndarray) -> bool:
    height, width, _ = frame.shape
    sidebar_pixel_width = int(width * SIDEBAR_WIDTH_PERCENT)
    left_sidebar = frame[:, :sidebar_pixel_width]
    right_sidebar = frame[:, width - sidebar_pixel_width:]
    center_start = sidebar_pixel_width + 50
    center_end = width - sidebar_pixel_width - 50
    center_region = frame[:, center_start:center_end]
    blur_left = _calculate_blurriness(left_sidebar)
    blur_right = _calculate_blurriness(right_sidebar)
    blur_center = _calculate_blurriness(center_region)
    if blur_center > BLUR_THRESHOLD and blur_left < BLUR_THRESHOLD and blur_right < BLUR_THRESHOLD:
        return True
    return False

# --- NEW: Helper function for split-screen detection ---
def is_split_screen(frame: np.ndarray) -> bool:
    """
    Detects if a frame is a split screen by finding a strong vertical edge profile.
    """
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Use Canny edge detection to find sharp edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Sum the edge pixels in each column
    column_edge_sums = np.sum(edges, axis=0)
    
    # We only search for a peak in the central part of the screen
    search_start = int(width * 0.2)
    search_end = int(width * 0.8)
    search_region = column_edge_sums[search_start:search_end]
    
    if len(search_region) > 0:
        # Calculate the average and peak edge concentration
        avg_edges = np.mean(search_region)
        peak_edges = np.max(search_region)
        
        # Avoid division by zero on blank frames
        if avg_edges == 0:
            return False
            
        # If the peak is significantly stronger than the average, it's a split screen
        if peak_edges / avg_edges > SPLIT_PEAK_FACTOR:
            return True
            
    return False
# ---------------------------------------------------


@app.post("/process-video/")
async def process_video_api(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    unique_id = uuid.uuid4()
    input_path = os.path.join(UPLOADS_DIR, f"{unique_id}_{file.filename}")
    output_path = os.path.join(PROCESSED_DIR, f"processed_{unique_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"Processing {input_path} with MoviePy...")

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        smoothed_center_x = None

        def process_frame(frame):
            nonlocal smoothed_center_x
            
            frame_height, frame_width, _ = frame.shape
            
            if smoothed_center_x is None:
                smoothed_center_x = frame_width / 2

            # --- REVISED LOGIC WITH CORRECT PRIORITY ---
            # 1. Check for TikTok/Reel format
            if is_vertical_reel(frame):
                target_center_x = frame_width / 2
                smoothed_center_x = frame_width / 2 # Reset smoother
            
            # 2. Check for Split-Screen format
            elif is_split_screen(frame):
                target_center_x = frame_width / 2
                smoothed_center_x = frame_width / 2 # Reset smoother
            
            # 3. Fallback to Person Recognition
            else:
                target_center_x = frame_width / 2 # Default center
                results = pose.process(frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    visible_landmarks_x = [lm.x * frame_width for lm in landmarks if lm.visibility > 0.5]
                    if visible_landmarks_x:
                        target_center_x = np.mean(visible_landmarks_x)
                
                # Apply smoothing only in this mode
                smoothed_center_x = (SMOOTHING_FACTOR * target_center_x) + ((1 - SMOOTHING_FACTOR) * smoothed_center_x)

            # Use the determined center (either absolute or smoothed) for cropping
            final_center_x = smoothed_center_x if not (is_vertical_reel(frame) or is_split_screen(frame)) else frame_width / 2
            
            # --- Cropping logic is now simpler ---
            vertical_height = frame_height
            vertical_width = int(vertical_height * 9 / 16)
            
            crop_left = int(final_center_x - vertical_width / 2)
            crop_right = int(final_center_x + vertical_width / 2)

            if crop_left < 0:
                crop_left = 0
                crop_right = min(vertical_width, frame_width)
            if crop_right > frame_width:
                crop_right = frame_width
                crop_left = max(0, frame_width - vertical_width)
            
            return frame[:, crop_left:crop_right]

        with VideoFileClip(input_path) as clip:
            processed_clip = clip.fl_image(process_frame)
            processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger='bar')

    print(f"Processing complete. Output at {output_path}")

    background_tasks.add_task(os.remove, input_path)
    background_tasks.add_task(os.remove, output_path)

    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")