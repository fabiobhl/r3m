import cv2
from cv2.typing import MatLike
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import math
import json

def process_video_chunk(video_path: Path, output_dir: Path, start_frame: int, end_frame: int, frame_size: tuple[int, int] = (224, 224)):
    """
    Extracts frames from video and saves them as images in the output directory.
    """
    
    # Open video file
    cap = cv2.VideoCapture(video_path.as_posix())
    
    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # resize frame
        frame = cv2.resize(frame, frame_size)
        
        # save frame to disk
        frame_filename = output_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)

        # update frame count
        frame_count += 1

    cap.release()

def extract_frames(video_path: Path, output_dir: Path, frame_size: tuple[int, int] = (224, 224)):
    """
    Extracts frames from video and saves them as images in the output directory.
    """

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video file
    cap = cv2.VideoCapture(video_path.as_posix())

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set the number of frames per chunk
    chunk_size = math.ceil(total_frames / os.cpu_count())

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for i in range(os.cpu_count()):
            start_frame = i * chunk_size
            end_frame = min((i + 1) * chunk_size, total_frames)
            futures.append(executor.submit(process_video_chunk, video_path, output_dir, start_frame, end_frame, frame_size))

        # Wait for all futures to complete
        for future in futures:
            future.result()

    cap.release()

def main(input_dir: str, output_dir: str, rel_clips_file: str):

    # convert input_dir and output_dir to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rel_clips_file = Path(rel_clips_file)

    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # load the clips manifest (json file)
    with open(rel_clips_file, 'r') as f:
        clips_manifest = json.load(f)

    # make sure that all clips in the manifest are present in the input directory
    for video_filename in clips_manifest:
        video_filename = video_filename + '.mp4'
        video_path = input_dir / video_filename
        if not video_path.exists():
            raise Exception(f"Video file {video_path} does not exist.")

    # Extract frames and organize them into folders
    for video_filename in tqdm(clips_manifest):
        video_filename = video_filename + '.mp4'
        video_path = input_dir / video_filename
        output_dir_video = output_dir / video_filename.split('.')[0]
        extract_frames(video_path, output_dir_video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process videos and create a dataset with manifest.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing video files.')
    parser.add_argument('--rel_clips_file', type=str, help='Path to the file containing the clips manifest.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory to save the processed dataset.')

    args = parser.parse_args()

    # count cpu cores and ask user to confirm
    print(f"Number of CPU cores: {os.cpu_count()}")
    print("Press any key to continue...")

    main(args.input_dir, args.output_dir, args.rel_clips_file)