import cv2
import h5py
import numpy as np

def extract_frames_to_compressed_hdf5(video_path: str, hdf5_path: str, compression='gzip'):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Create a group for frames
            frames_group = hdf5_file.create_group('frames')
            
            frame_idx = 0
            while cap.isOpened():
                # read in frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # resize frame
                frame = cv2.resize(frame, (224, 224))                

                # Encode the frame as a JPEG image
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                
                # Store each frame as a separate dataset within the frames group
                frames_group.create_dataset(str(frame_idx), data=np.frombuffer(encimg.tobytes(), dtype=np.uint8), compression=compression)
                
                frame_idx += 1
        
        cap.release()
    
    except Exception as e:
        print(f"Failed to store frame in HDF5 file for video path: {video_path} \n with error: {e}")

def load_frame_from_compressed_hdf5(hdf5_path, frame_idx):
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Get the 'frames' group
        frames_group = hdf5_file['frames']
        
        # Access the random frame dataset
        random_frame_data = frames_group[str(frame_idx)][:]
        
        # Decode the JPEG compressed frame
        random_frame = cv2.imdecode(np.frombuffer(random_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
    return random_frame

