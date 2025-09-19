import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
import cv2

app = FastAPI()
mp_pose = mp.solutions.pose
SMOOTHING_FACTOR = 0.05
UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"

BLUR_THRESHOLD = 100 
SIDEBAR_WIDTH_PERCENT = 0.25 

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


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

def detect_clip_borders(frame: np.ndarray, cache: dict) -> list:
  if "borders" in cache:
    return cache["borders"]

  height, width, _ = frame.shape

  thumbnail_width = 150
  resized = cv2.resize(frame, (thumbnail_width, int(height * thumbnail_width / width)))
  
  diffs = []
  for i in range(1, thumbnail_width):
    col_prev = resized[:, i-1]
    col_curr = resized[:, i]

    diff = np.sum(np.abs(col_prev.astype("float") - col_curr.astype("float")))
    diffs.append(diff)

  if not diffs:
    cache["borders"] = [0, width]
    return [0, width]
      
  max_diff_index = np.argmax(diffs)

  border_x = int((max_diff_index + 1) * (width / thumbnail_width))

  if border_x > width * 0.1 and border_x < width * 0.9:
    print(f"Content-based border detected at x={border_x}")
    borders = sorted([0, border_x, width])
    cache["borders"] = borders
    return borders
  else:
    print("No significant content-based border found. Using full frame.")
    cache["borders"] = [0, width]
    return [0, width]

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
    frame_cache = {}

    def process_frame(frame):
      nonlocal smoothed_center_x
      
      frame_height, frame_width, _ = frame.shape
      
      if smoothed_center_x is None:
        smoothed_center_x = frame_width / 2

      borders = detect_clip_borders(frame, frame_cache)

      if is_vertical_reel(frame):
        target_center_x = frame_width / 2
      else:
        target_center_x = frame_width / 2
        results = pose.process(frame)
        if results.pose_landmarks:
          landmarks = results.pose_landmarks.landmark
          visible_landmarks_x = [lm.x * frame_width for lm in landmarks if lm.visibility > 0.5]
          if visible_landmarks_x:
            target_center_x = np.mean(visible_landmarks_x)

      smoothed_center_x = (SMOOTHING_FACTOR * target_center_x) + ((1 - SMOOTHING_FACTOR) * smoothed_center_x)

      clip_left_boundary = 0
      clip_right_boundary = frame_width
      if len(borders) > 2:
        for i in range(len(borders) - 1):
          if borders[i] <= smoothed_center_x < borders[i+1]:
            clip_left_boundary = borders[i]
            clip_right_boundary = borders[i+1]
            break

      vertical_height = frame_height
      vertical_width = int(vertical_height * 9 / 16)

      if vertical_width > (clip_right_boundary - clip_left_boundary):
          vertical_width = clip_right_boundary - clip_left_boundary

      crop_left = int(smoothed_center_x - vertical_width / 2)
      crop_right = int(smoothed_center_x + vertical_width / 2)

      if crop_left < clip_left_boundary:
        crop_left = clip_left_boundary
        crop_right = clip_left_boundary + vertical_width
      if crop_right > clip_right_boundary:
        crop_right = clip_right_boundary
        crop_left = clip_right_boundary - vertical_width

      crop_left = max(0, int(crop_left))
      crop_right = min(frame_width, int(crop_right))
      
      return frame[:, crop_left:crop_right]

    with VideoFileClip(input_path) as clip:
      processed_clip = clip.fl_image(process_frame)
      processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger='bar')

  print(f"Processing complete. Output at {output_path}")

  background_tasks.add_task(os.remove, input_path)
  background_tasks.add_task(os.remove, output_path)

  return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")