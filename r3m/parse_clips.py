import cv2
from cv2.typing import MatLike
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
import json
from r3m.utils.clip_processing import extract_frames_to_compressed_hdf5


def main(input_dir: str, output_dir: str, relevant_clips_file: str, max_workers: int):

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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        with tqdm(total=len(video_paths)) as pbar:
            
            # Submit the first set of tasks
            for _ in range(max_workers):
                if not video_paths:
                    break
                video_path = video_paths.pop(0)
                hdf5_file_path = hdf5_file_paths.pop(0)
                futures.append(executor.submit(extract_frames_to_compressed_hdf5, video_path, hdf5_file_path))
            
            # As futures complete, submit new tasks if any
            for future in as_completed(futures):
                # Wait for the current future to complete
                future.result()  
                # Update progress bar
                pbar.update(1)  
                # Check if there are remaining tasks to submit
                if video_paths:
                    video_path = video_paths.pop(0)
                    hdf5_file_path = hdf5_file_paths.pop(0)
                    futures.append(executor.submit(extract_frames_to_compressed_hdf5, video_path, hdf5_file_path))
                
    print("Finished processing videos.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process videos and create a dataset with manifest.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing video files.')
    parser.add_argument('--rel_clips_file', type=str, help='Path to the file containing the clips manifest.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory to save the processed dataset.')
    parser.add_argument('--max_workers', type=int, help='Number of workers to use for processing videos. Default is the number of CPU cores.')

    args = parser.parse_args()

    # count cpu cores and ask user to confirm
    print(f"Number of CPU cores: {os.cpu_count()}")
    input("Press any key to continue...")

    main(args.input_dir, args.output_dir, args.rel_clips_file, args.max_workers)