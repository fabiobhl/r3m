import json
from tqdm import tqdm
from pathlib import Path
import os
import argparse

def main(input_path: str, output_path: str):

    # convert input_path and output_path to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    # make sure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    with open(input_path) as f:
        data = json.load(f)

    relevant_clips = []
    for video in tqdm(data["videos"]):
        for clip in video["annotated_intervals"]:
            relevant = False
            for annotation in clip["narrated_actions"]:
                if annotation["is_rejected"]:
                    continue
                else:
                    relevant = True
                    break

            if relevant:
                relevant_clips.append(clip["clip_uid"])

    with open(output_path / "relevant_clips.json", "w") as f:
        json.dump(relevant_clips, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find all relevant clips from the fho_main.json file.')
    parser.add_argument('--file_path', type=str, help='Path to the input file')
    parser.add_argument('--output_path', type=str, help='Path to the output directory to save the processed list.')

    args = parser.parse_args()

    main(args.file_path, args.output_path)