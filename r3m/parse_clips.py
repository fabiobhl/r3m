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
from r3m.utils.clip_processing import extract_frames_to_compressed_hdf5

def process_video_list(video_paths: list[str], hdf5_file_paths: list[str], tqdm_enabled=False):
    if tqdm_enabled:
        pbar = tqdm(total=len(video_paths), desc="Extracting and compressing videos")
    else:
        pbar = None
    
    for video_path, hdf5_file_path in zip(video_paths, hdf5_file_paths):
        extract_frames_to_compressed_hdf5(video_path, hdf5_file_path)
        if pbar is not None:
            pbar.update(1)
    
    if pbar is not None:
        pbar.close()

def main(input_dir: str, output_dir: str, relevant_clips_file: str):

    # convert input_dir and output_dir to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    relevant_clips_file = Path(relevant_clips_file)

    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load the relevant clips file
    with open(relevant_clips_file, 'r') as f:
        relevant_clips = json.load(f)

    # make sure that all clips are present in the input directory
    for video_filename in relevant_clips:
        video_filename = video_filename + '.mp4'
        video_path = input_dir / video_filename
        if not video_path.exists():
            raise Exception(f"Video file {video_path} does not exist.")

    # Extract frames and organize them into folders
    video_paths = []
    hdf5_file_paths = []
    for video_filename in relevant_clips:
        video_filename = video_filename + '.mp4'
        video_path = input_dir / video_filename
        hdf5_file_path = output_dir / f"{video_filename.split('.')[0]}.hdf5"
        video_paths.append(video_path)
        hdf5_file_paths.append(hdf5_file_path)
        
    
    # Process the videos in parallel    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        
        chunk_size = math.ceil(len(video_paths) / os.cpu_count())

        for i in range(os.cpu_count()):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(video_paths))
            if i == 0:
                tqdm_enabled = True
            else:
                tqdm_enabled = False
            futures.append(executor.submit(process_video_list, video_paths[start_idx:end_idx], hdf5_file_paths[start_idx:end_idx], tqdm_enabled))

        # Wait for all futures to complete
        for future in futures:
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process videos and create a dataset with manifest.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing video files.')
    parser.add_argument('--rel_clips_file', type=str, help='Path to the file containing the clips manifest.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory to save the processed dataset.')

    args = parser.parse_args()

    # count cpu cores and ask user to confirm
    print(f"Number of CPU cores: {os.cpu_count()}")
    input("Press any key to continue...")

    main(args.input_dir, args.output_dir, args.rel_clips_file)