import cv2
import os


def extract_frames(video_path, output_dir, fps=5):
    """Extract frames dari video dengan interval tertentu"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # interval = int(video_fps / fps)  # ambil N frame per detik
    interval = 7 # sama dengan sample_freq pphuman config

    frame_idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_idx:05d}.jpg", frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"Tersimpan {saved} frames ke {output_dir}")


extract_frames("storage/private/videos/fight_0990.mpeg", "storage/private/frames/fight_0990_2/input/")
