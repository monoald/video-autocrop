# main.py

import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
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


def create_vertical_video(input_path: str, output_path: str):
  with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
      raise IOError(f"Could not open video file {input_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    
    temp_silent_output_path = os.path.join(PROCESSED_DIR, f"silent_{uuid.uuid4()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_silent_output_path, fourcc, fps, (vertical_width, vertical_height))

    smoothed_center_x = original_width / 2

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(rgb_frame)
      target_center_x = original_width / 2
      
      if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        visible_landmarks_x = [lm.x * original_width for lm in landmarks if lm.visibility > 0.5]
        if visible_landmarks_x:
          target_center_x = np.mean(visible_landmarks_x)

      smoothed_center_x = (SMOOTHING_FACTOR * target_center_x) + ((1 - SMOOTHING_FACTOR) * smoothed_center_x)
      
      crop_left = int(smoothed_center_x - vertical_width / 2)
      crop_right = int(smoothed_center_x + vertical_width / 2)

      if crop_left < 0:
        crop_left = 0
        crop_right = vertical_width
      if crop_right > original_width:
        crop_right = original_width
        crop_left = original_width - vertical_width
    
      cropped_frame = frame[:, crop_left:crop_right]
      out.write(cropped_frame)

    cap.release()
    out.release()

    original_video_clip = VideoFileClip(input_path)
    if original_video_clip.audio:
      cropped_video_clip = VideoFileClip(temp_silent_output_path)
      final_clip = cropped_video_clip.set_audio(original_video_clip.audio)
      final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
      cropped_video_clip.close()
      final_clip.close()
    else:
      os.rename(temp_silent_output_path, output_path)
        
    original_video_clip.close()

    if os.path.exists(temp_silent_output_path):
      os.remove(temp_silent_output_path)


@app.post("/process-video/")
async def process_video_api(file: UploadFile = File(...)):
    unique_id = uuid.uuid4()
    input_path = os.path.join(UPLOADS_DIR, f"{unique_id}_{file.filename}")
    output_path = os.path.join(PROCESSED_DIR, f"processed_{unique_id}.mp4")

    try:
      with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

      print(f"Processing {input_path}...")
      create_vertical_video(input_path, output_path)
      print(f"Processing complete. Output at {output_path}")

      return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")

    finally:
      if os.path.exists(input_path):
        os.remove(input_path)
      if os.path.exists(output_path):
        os.remove(output_path)