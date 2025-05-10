from ast import parse
import os
import re
import numpy as np
import random
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
import csv
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder
from logger.utils import traverse_dir
import concurrent.futures

# disable gc
import gc

gc.disable()


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set",
    )
    parser.add_argument(
        "-f",
        "--filelist",
        type=str,
        default=None,
        required=False,
        help="path to a text file containing the list of files to process",
    )
    parser.add_argument(
        "-r", "--root_path", type=str, required=True, help="root path of the dataset"
    )
    parser.add_argument(
        "--autopick",
        action="store_true",
        help="automatically pick an unprocessed csv from filelist directory",
    )
    return parser.parse_args(args=args, namespace=namespace)


def find_first_unprocessed_csv(filelist_dir):
    for file in sorted(os.listdir(filelist_dir)):
        if file.endswith(".csv") and not file.startswith("completed_"):
            return os.path.join(filelist_dir, file)
    return None


def preprocess(
    path,
    f0_extractor,
    volume_extractor,
    mel_extractor,
    units_encoder,
    sample_rate,
    hop_size,
    device="cuda",
    use_pitch_aug=False,
    extensions=["wav"],
    filelist_path=None,
):
    path_srcdir = os.path.join(path, "audio")
    path_features1dir = os.path.join(path, "features1")
    path_features2dir = os.path.join(path, "features2")
    path_skipdir = os.path.join(path, "skip")

    filelist = []
    print("FileList Path", filelist_path)
    if filelist_path is not None and os.path.exists(filelist_path):
        if filelist_path.endswith(".csv"):
            with open(filelist_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                filelist = [(int(row[0]), row[1]) for row in reader if len(row) >= 2]
            print(f"Loaded {len(filelist)} files from CSV: {filelist_path}")
        else:
            with open(filelist_path, "r", encoding="utf-8") as f:
                filelist = [(0, line.strip()) for line in f.readlines() if line.strip()]
            print(f"Loaded {len(filelist)} files from text file: {filelist_path}")
    else:
        files = traverse_dir(
            path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True
        )
        filelist = [(0, file) for file in files]

    def process(file_tuple):
        spk_id, file = file_tuple
        binfile = file + ".npy"
        path_srcfile = os.path.join(path_srcdir, file)
        path_features1file = os.path.join(path_features1dir, binfile)
        path_features2file = os.path.join(path_features2dir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)

        if os.path.exists(path_features1file) and os.path.exists(path_features2file):
            print(f"Skip {path_features1file} and {path_features2file} already exists.")
            return

        # path_srcfile 不存在则 skip

        if not os.path.exists(path_srcfile):
            print(f"Skip {path_srcfile} does not exist.")
            return

        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)

        volume = volume_extractor.extract(audio)

        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to("cpu").numpy()

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            keyshift = random.uniform(-5, 5) if use_pitch_aug else 0

            aug_mel_t = mel_extractor.extract(
                audio_t * (10**log10_vol_shift), sample_rate, keyshift=keyshift
            )
            aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
            aug_vol = volume_extractor.extract(audio * (10**log10_vol_shift))

        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to("cpu").numpy()

        f0 = f0_extractor.extract(audio, uv_interp=False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            mel_len = mel.shape[0] if mel_extractor is not None else 0
            aug_mel_len = aug_mel.shape[0] if mel_extractor is not None else 0
            units_len = units.shape[0]
            f0_len = len(f0)
            volume_len = len(volume)
            aug_vol_len = len(aug_vol) if mel_extractor is not None else 0

            frame_len = min(
                mel_len, aug_mel_len, units_len, f0_len, volume_len, aug_vol_len
            )

            features1 = {
                "frame_len": frame_len,
                "f0": f0,
                "volume": volume,
                "aug_vol": aug_vol if mel_extractor is not None else None,
                "spk_id": spk_id,
                "pitch_aug": keyshift if use_pitch_aug else 0,
            }
            features2 = {
                "mel": mel if mel_extractor is not None else None,
                "aug_mel": aug_mel if mel_extractor is not None else None,
                "units": units,
            }

            os.makedirs(os.path.dirname(path_features1file), exist_ok=True)
            os.makedirs(os.path.dirname(path_features2file), exist_ok=True)
            np.save(path_features1file, features1)
            np.save(path_features2file, features2)
        else:
            print("\n[Error] F0 extraction failed: " + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print("This file has been moved to " + path_skipfile)

    print("Preprocess the audio clips in :", path_srcdir)
    if filelist_path is not None:
        print(f"Using file list from: {filelist_path}")

    for file_tuple in tqdm(filelist, total=len(filelist)):
        process(file_tuple)

    if filelist_path is not None and os.path.exists(filelist_path):
        dir_name = os.path.dirname(filelist_path)
        base_name = os.path.basename(filelist_path)
        new_name = os.path.join(dir_name, f"completed_{base_name}")
        os.rename(filelist_path, new_name)
        print(f"Renamed filelist to: {new_name}")


if __name__ == "__main__":
    cmd = parse_args()
    device = cmd.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if cmd.autopick and cmd.filelist is None:
        filelist_dir = "filelist"
        picked = find_first_unprocessed_csv(filelist_dir)
        if picked:
            cmd.filelist = picked
            print(f"[AutoPick] Selected filelist: {picked}")
        else:
            print("[AutoPick] No unprocessed CSV file found in filelist directory.")
            exit(1)

    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions

    f0_extractor = F0_Extractor(
        args.data.f0_extractor,
        args.data.sampling_rate,
        args.data.block_size,
        args.data.f0_min,
        args.data.f0_max,
    )

    volume_extractor = Volume_Extractor(
        args.data.block_size, args.data.volume_smooth_size
    )

    mel_extractor = None
    use_pitch_aug = False
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    if (
        mel_extractor.vocoder_sample_rate != sample_rate
        or mel_extractor.vocoder_hop_size != hop_size
    ):
        mel_extractor = None
        print("Unmatch vocoder parameters, mel extraction is ignored!")
    elif args.model.use_pitch_aug:
        use_pitch_aug = True

    if args.data.encoder == "cnhubertsoftfish":
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10

    units_encoder = Units_Encoder(
        args.data.encoder,
        args.data.encoder_ckpt,
        args.data.encoder_sample_rate,
        args.data.encoder_hop_size,
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device,
    )

    preprocess(
        cmd.root_path,
        f0_extractor,
        volume_extractor,
        mel_extractor,
        units_encoder,
        sample_rate,
        hop_size,
        device=device,
        use_pitch_aug=use_pitch_aug,
        extensions=extensions,
        filelist_path=cmd.filelist,
    )
