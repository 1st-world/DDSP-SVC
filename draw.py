import argparse
import random
from tqdm import tqdm
import os
import shutil
import soundfile as sf

WAV_MIN_LENGTH = 2  # The minimum duration of wav files
SAMPLE_MIN = 2      # The lower limit of the number of files to be extracted
SAMPLE_MAX = 10     # The upper limit of the number of files to be extracted

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    root_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default=root_dir + "/data/train/audio",
        help="Directory where contains train dataset"
    )
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=root_dir + "/data/val/audio",
        help="Directory where contains validate dataset"
    )
    parser.add_argument(
        "-r",
        "--sample_rate",
        type=float,
        default=1,
        help="The percentage of files to be extracted | Default: 1"
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        required=False,
        nargs="*",
        default=["wav", "flac"],
        help="List of using file extensions, e.g. -e wav flac ..."
    )
    return parser.parse_args(args=args, namespace=namespace)

# Check if the WAV file duration is greater than WAV_MIN_LENGTH
def check_duration(wav_file):
    with sf.SoundFile(wav_file) as f:
        duration = f.frames / float(f.samplerate)
    return duration > WAV_MIN_LENGTH

def get_unique_filename(dst_dir, filename):
    """ Generate a unique filename by appending a number if the file already exists
    :param dst_dir(str): Destination directory
    :param filename(str): Original filename
    :return(str): Unique filename
    """
    name, ext = os.path.splitext(filename)
    count = 1
    new_filename = filename
    while os.path.exists(os.path.join(dst_dir, new_filename)):
        new_filename = f"{name}_{count}{ext}"
        count += 1
    return new_filename

# Split data from src_dir to dst_dir based on the given ratio and extensions
def split_data(src_dir, dst_dir, ratio, extensions):
    # Create dst_dir if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Collect all files with specified extensions
    all_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if any(file.endswith(f".{ext}") for ext in extensions):
                all_files.append(os.path.join(root, file))
    
    # Check if any files were found
    if not all_files:
        print(f"Error: No files with specified extensions found in {src_dir}")
        return
    
    # Calculate number of files to move, bounded by SAMPLE_MIN and SAMPLE_MAX
    num_files = int(len(all_files) * ratio)
    num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))
    
    # Randomly select files to move
    selected_files = random.sample(all_files, num_files)
    
    # Move selected files to the root of the dst_dir
    with tqdm(total=num_files, desc="Moving... ") as pbar:
        for src_file in selected_files:
            # Check file duration (assumes audio files)
            if not check_duration(src_file):
                print(f"Skipped {src_file} because its duration is less than 2 seconds.")
                continue
            # Generate a unique filename for the dst
            base_filename = os.path.basename(src_file)
            unique_filename = get_unique_filename(dst_dir, base_filename)
            dst_file = os.path.join(dst_dir, unique_filename)
            # Move file and handle potential errors
            try:
                shutil.move(src_file, dst_file)
                pbar.update(1)
            except Exception as e:
                print(f"Failed to move {src_file}: {str(e)}")

# Main function to parse commands and call split_data()
def main(cmd = None):
    if cmd is None:
        cmd = parse_args()
    src_dir = cmd.train
    dst_dir = cmd.val
    ratio = cmd.sample_rate / 100
    extensions = cmd.extensions
    split_data(src_dir, dst_dir, ratio, extensions)

if __name__ == "__main__":
    main()
