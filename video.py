
import os
import cv2

video_dir = "paired_frames/-0.2000000000000003_25"
output_path = "paired_video.mp4"
fps = 5

# Get all image files and sort them naturally

frames = [f for f in os.listdir(video_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
frames = sorted(frames)

if not frames:
	raise ValueError("No image frames found in directory.")

# Read the first frame to get the size
first_frame = cv2.imread(os.path.join(video_dir, frames[0]))
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame_name in frames:
	frame_path = os.path.join(video_dir, frame_name)
	img = cv2.imread(frame_path)
	if img is None:
		print(f"Warning: Could not read {frame_path}, skipping.")
		continue
	video.write(img)

video.release()
print(f"Video saved to {output_path}")