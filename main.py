import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip

app = FastAPI()
mp_pose = mp.solutions.pose
SMOOTHING_FACTOR = 0.05
UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

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
      if smoothed_center_x is None:
        smoothed_center_x = frame.shape[1] / 2
      results = pose.process(frame)
      target_center_x = frame.shape[1] / 2
      if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        visible_landmarks_x = [lm.x * frame.shape[1] for lm in landmarks if lm.visibility > 0.5]
        if visible_landmarks_x:
          target_center_x = np.mean(visible_landmarks_x)
      smoothed_center_x = (SMOOTHING_FACTOR * target_center_x) + ((1 - SMOOTHING_FACTOR) * smoothed_center_x)
      vertical_height = frame.shape[0]
      vertical_width = int(vertical_height * 9 / 16)
      crop_left = int(smoothed_center_x - vertical_width / 2)
      crop_right = int(smoothed_center_x + vertical_width / 2)
      if crop_left < 0:
        crop_left = 0
        crop_right = vertical_width
      if crop_right > frame.shape[1]:
        crop_right = frame.shape[1]
        crop_left = frame.shape[1] - vertical_width
      return frame[:, crop_left:crop_right]

    with VideoFileClip(input_path) as clip:
      processed_clip = clip.fl_image(process_frame)
      processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger='bar')

  print(f"Processing complete. Output at {output_path}")

  background_tasks.add_task(os.remove, input_path)
  background_tasks.add_task(os.remove, output_path)

  return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")