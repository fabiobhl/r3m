import json
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import pandas as pd

def main(rel_clips_file: str, main_json_file: str, output_dir: str):
    
    output_dir = Path(output_dir)
    
    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load the relevant clips list (json file)
    with open(rel_clips_file, 'r') as f:
        relevant_clips = json.load(f)
    
    # load the main json file
    with open(main_json_file, 'r') as f:
        data = json.load(f)

    manifest_list = []
    for video in tqdm(data["videos"]):
        for clip in video["annotated_intervals"]:
            if clip["clip_uid"] in relevant_clips:
                for annotation in clip["narrated_actions"]:
                    if not annotation["is_rejected"]:
                        manifest_list.append([
                            str(clip["clip_uid"]),
                            annotation["clip_start_frame"],
                            annotation["clip_end_frame"],
                            annotation["narration_text"],
                        ])
                        
    df = pd.DataFrame(manifest_list, columns=["clip_uid", "clip_start_frame", "clip_end_frame", "narration_text"])
    
    df.to_csv(output_dir / "manifest.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find all relevant clips from the fho_main.json file.')
    parser.add_argument('--rel_clips_file', type=str)
    parser.add_argument('--main_json_file', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    main(args.rel_clips_file, args.main_json_file, args.output_dir)
    
    """
    --rel_clips_file=/home/fabio/git/r3m/r3m/utils/relevant_clips.json
    --main_json_file=/home/fabio/Documents/semester_project/Ego4d/v2/annotations/fho_main.json
    --output_dir=/home/fabio/Documents/semester_project/Ego4d/v2
    """