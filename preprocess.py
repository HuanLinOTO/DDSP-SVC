import os
import numpy as np
import random
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
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
    return parser.parse_args(args=args, namespace=namespace)


def np_feature_to_unsqueeze_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().unsqueeze(-1)


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
):
    path_srcdir = os.path.join(path, "audio")
    path_bindir = os.path.join(path, "features")
    path_skipdir = os.path.join(path, "skip")

    # list files
    filelist = traverse_dir(
        path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True
    )

    # pitch augmentation dictionary
    # run
    def process(file):
        binfile1 = file + ".1.pt"
        binfile2 = file + ".2.pt"

        path_srcfile = os.path.join(path_srcdir, file)

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

            os.makedirs(
                os.path.dirname(os.path.join(path_bindir, binfile1)), exist_ok=True
            )

            f0 = np_feature_to_unsqueeze_tensor(f0)
            units = torch.from_numpy(units)
            volume = np_feature_to_unsqueeze_tensor(volume)
            aug_vol = np_feature_to_unsqueeze_tensor(aug_vol)

            torch.save(
                dict(
                    units=units,
                    mel=mel,
                    aug_mel=aug_mel,
                    keyshift=keyshift,
                ),
                os.path.join(path_bindir, binfile1),
            )

            torch.save(
                dict(
                    f0=f0,
                    volume=volume,
                    aug_vol=aug_vol,
                    frame_len=min(
                        mel.shape[0],
                        units.shape[0],
                        f0.shape[0],
                        volume.shape[0],
                    ),
                ),
                os.path.join(path_bindir, binfile2),
            )

        else:
            path_skipfile = os.path.join(path_skipdir, file)
            print("\n[Error] F0 extraction failed: " + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print("This file has been moved to " + path_skipfile)

    print("Preprocess the audio clips in :", path_srcdir)

    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)


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
        args.data.train_path,
        f0_extractor,
        volume_extractor,
        mel_extractor,
        units_encoder,
        sample_rate,
        hop_size,
        device=device,
        use_pitch_aug=use_pitch_aug,
        extensions=extensions,
    )

    # preprocess validation set
    preprocess(
        args.data.valid_path,
        f0_extractor,
        volume_extractor,
        mel_extractor,
        units_encoder,
        sample_rate,
        hop_size,
        device=device,
        use_pitch_aug=False,
        extensions=extensions,
    )
