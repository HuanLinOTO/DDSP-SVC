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
        "-r",
        "--root_path",
        type=str,
        required=True,
        help="root path of the dataset",
    )
    return parser.parse_args(args=args, namespace=namespace)


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

    # list files
    filelist = []
    if filelist_path is not None and os.path.exists(filelist_path):
        # 检查文件扩展名判断是CSV还是TXT
        if filelist_path.endswith('.csv'):
            with open(filelist_path, "r", encoding="utf-8", newline='') as f:
                reader = csv.reader(f)
                filelist = [(int(row[0]), row[1]) for row in reader if len(row) >= 2]
            print(f"Loaded {len(filelist)} files from CSV: {filelist_path}")
        else:
            # 向后兼容旧格式
            with open(filelist_path, "r", encoding="utf-8") as f:
                filelist = [(0, line.strip()) for line in f.readlines() if line.strip()]
            print(f"Loaded {len(filelist)} files from text file: {filelist_path}")
    else:
        files = traverse_dir(
            path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True
        )
        # 如果没有提供文件列表，则使用默认spk_id为0
        filelist = [(0, file) for file in files]

    # run
    def process(file_tuple):
        spk_id, file = file_tuple
        binfile = file + ".npy"
        path_srcfile = os.path.join(path_srcdir, file)
        path_features1file = os.path.join(path_features1dir, binfile)
        path_features2file = os.path.join(path_features2dir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)

        # 如果 features1/2 已经存在，则跳过
        if os.path.exists(path_features1file) and os.path.exists(path_features2file):
            print(f"Skip {path_features1file} and {path_features2file} already exists.")
            return

        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        # extract volume
        volume = volume_extractor.extract(audio)

        # extract mel and volume augmentaion
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to("cpu").numpy()

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0

            aug_mel_t = mel_extractor.extract(
                audio_t * (10**log10_vol_shift), sample_rate, keyshift=keyshift
            )
            aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
            aug_vol = volume_extractor.extract(audio * (10**log10_vol_shift))

        # units encode
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to("cpu").numpy()

        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp=False)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # Calculate lengths for alignment
            mel_len = mel.shape[0] if mel_extractor is not None else 0
            aug_mel_len = aug_mel.shape[0] if mel_extractor is not None else 0
            units_len = units.shape[0]
            f0_len = len(f0)
            volume_len = len(volume)
            aug_vol_len = len(aug_vol) if mel_extractor is not None else 0

            frame_len = min(
                mel_len, aug_mel_len, units_len, f0_len, volume_len, aug_vol_len
            )

            # 现在使用传入的spk_id，而不是从文件路径解析
            features1 = {
                "frame_len": frame_len,
                "f0": f0,
                "volume": volume,
                "aug_vol": aug_vol if mel_extractor is not None else None,
                "spk_id": spk_id,  # 使用CSV中提供的spk_id
                "pitch_aug": keyshift if use_pitch_aug else 0,
            }

            features2 = {
                "mel": mel if mel_extractor is not None else None,
                "aug_mel": aug_mel if mel_extractor is not None else None,
                "units": units,
            }

            # make sure the directory exists
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

    # single process
    for file_tuple in tqdm(filelist, total=len(filelist)):
        process(file_tuple)

    # if mel_extractor is not None:
    #     path_pitchaugdict = os.path.join(path, "pitch_aug_dict.npy")
    #     np.save(path_pitchaugdict, pitch_aug_dict)
    # multi-process (have bugs)
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    """


if __name__ == "__main__":
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    extensions = args.data.extensions

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
        args.data.f0_extractor,
        args.data.sampling_rate,
        args.data.block_size,
        args.data.f0_min,
        args.data.f0_max,
    )

    # initialize volume extractor
    volume_extractor = Volume_Extractor(
        args.data.block_size, args.data.volume_smooth_size
    )

    # initialize mel extractor
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

    # initialize units encoder
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

    # preprocess training set
    preprocess(
        # args.data.train_path,
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