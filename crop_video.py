import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
import argparse

SMOOTHING_FACTOR = 0.05

def create_vertical_video(input_path, output_path):
    print("Starting video processing...")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Vertical crop resolution: {vertical_width}x{vertical_height}")

    temp_silent_output_path = "temp_silent_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent_output_path, fourcc, fps, (vertical_width, vertical_height))

    smoothed_center_x = original_width / 2

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        target_center_x = original_width / 2
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            x_coords = [lm.x * original_width for lm in landmarks]

            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            visible_landmarks_x = [
                lm.x * original_width for lm in [nose, left_shoulder, right_shoulder] if lm.visibility > 0.5
            ]
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
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    print("Releasing video resources...")
    cap.release()
    out.release()
    pose.close()

    print("Adding original audio to the cropped video...")
    original_video_clip = VideoFileClip(input_path)
    cropped_video_clip = VideoFileClip(temp_silent_output_path)
    
    final_clip = cropped_video_clip.set_audio(original_video_clip.audio)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="mp3")

    print(f"Processing complete! Vertical video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vertical video by tracking a person.")
    parser.add_argument("--input", required=True, help="Path to the input horizontal video file.")
    parser.add_argument("--output", required=True, help="Path to save the output vertical video file.")
    
    args = parser.parse_args()
    
    create_vertical_video(args.input, args.output)