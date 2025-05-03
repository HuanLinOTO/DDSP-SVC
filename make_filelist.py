import os
import argparse
import csv
from tqdm import tqdm
from logger.utils import traverse_dir


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="data",
        help="base path to the data directory (default: data)",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=["wav"],
        help="file extensions to include (default: wav)",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=2000,
        help="number of files per chunk (default: 2000)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="filelist",
        help="output directory for file lists (default: filelist)",
    )
    return parser.parse_args(args=args, namespace=namespace)

def process_directory(
    data_path, dir_name, output_dir, extensions=["wav"], chunk_size=2000
):
    """Process a specific directory and generate file lists."""
    path = os.path.join(data_path, dir_name)
    if not os.path.exists(path):
        print(f"Directory {path} does not exist, skipping.")
        return

    # List audio files
    path_srcdir = os.path.join(path, "audio")
    if not os.path.exists(path_srcdir):
        print(f"Audio directory {path_srcdir} does not exist, skipping.")
        return
    
    # Get all subdirectories in audio directory
    subdirs = [d for d in os.listdir(path_srcdir) 
               if os.path.isdir(os.path.join(path_srcdir, d))]
    subdirs.sort()  # Sort to ensure consistent speaker IDs
    
    # Create speaker ID mapping (starting from 1)
    spk_mapping = {subdir: idx+1 for idx, subdir in enumerate(subdirs)}
    
    # Collect all files with their speaker IDs
    all_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(path_srcdir, subdir)
        files = traverse_dir(
            subdir_path, extensions=extensions, is_pure=False, is_sort=True, is_ext=True
        )
        
        # Convert to relative paths and add speaker ID
        rel_files = []
        for file in files:
            rel_path = os.path.relpath(file, path_srcdir)
            spk_id = spk_mapping[subdir]
            rel_files.append((spk_id, rel_path))
        
        all_files.extend(rel_files)
    
    # Split files into chunks
    total_files = len(all_files)
    if total_files == 0:
        print(all_files)
        print(f"No files found in {path_srcdir}, skipping.")
        return

    num_chunks = (total_files + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Found {total_files} files in {dir_name}, creating {num_chunks} chunks...")
    print(f"Assigned {len(subdirs)} speaker IDs based on directory order.")

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        chunk_files = all_files[start_idx:end_idx]

        # Create output file (now as CSV)
        output_file = os.path.join(output_dir, f"{dir_name}_chunk_{i + 1:03d}.csv")

        with open(output_file, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            for spk_id, file_path in chunk_files:
                writer.writerow([spk_id, file_path])

        print(f"Created {output_file} with {len(chunk_files)} files")


def make_filelist(base_path, output_dir, extensions=["wav"], chunk_size=2000):
    """Generate file lists from train and val directories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process train directory
    process_directory(base_path, "train", output_dir, extensions, chunk_size)

    # Process val directory
    process_directory(base_path, "val", output_dir, extensions, chunk_size)


if __name__ == "__main__":
    # Parse commands
    cmd = parse_args()

    # Generate file lists
    make_filelist(
        cmd.path, cmd.output_dir, extensions=cmd.extensions, chunk_size=cmd.chunk_size
    )

    print("File lists generation completed!")
