import os
import random
import re
import time
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


# def get_npy_shape(file_path):
#     with open(file_path, "rb") as f:
#         version = np.lib.format.read_magic(f)
#         if version == (1, 0):
#             shape = np.lib.format.read_array_header_1_0(f)[0]
#         elif version == (2, 0):
#             shape = np.lib.format.read_array_header_2_0(f)[0]
#         else:
#             raise ValueError("Unsupported .npy file version")
#     return shape


def traverse_dir(
    root_dir,
    extensions,
    amount=None,
    str_include=None,
    str_exclude=None,
    is_pure=False,
    is_sort=False,
    is_ext=True,
):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1 :] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split(".")[-1]
                    pure_path = pure_path[: -(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True,
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device == "cpu" else 0,
        persistent_workers=(args.train.num_workers > 0)
        if args.train.cache_device == "cpu"
        else False,
        pin_memory=True if args.train.cache_device == "cpu" else False,
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    return loader_train, loader_valid


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=["wav"],
        n_spk=1,
        device="cpu",
        fp16=False,
        use_aug=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, "audio"),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True,
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug

        self.device = device

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        # data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        # if data_buffer["frame_len"] < self.crop_len:
        return self.get_data(name_ext)
        # return _
        # return self.__getitem__((file_idx + 1) % len(self.paths))

        # get item

    def get_data(self, name_ext):
        features = np.load(os.path.join(self.path_root, "features", name_ext) + ".npz")

        aug_flag = random.choice([True, False]) and self.use_aug

        mel_key = "aug_mel" if aug_flag else "mel"
        mel = features.get(mel_key)

        units = features.get("units")

        f0 = features.get("f0")
        f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(self.device)

        vol_key = "aug_vol" if aug_flag else "volume"
        volume = features.get(vol_key)
        volume = torch.from_numpy(volume).float().unsqueeze(-1).to(self.device)

        frame_len = min(
            mel.shape[0],
            units.shape[0],
            f0.shape[0],
            volume.shape[0],
        )

        name = os.path.splitext(name_ext)[0]
        start_frame = (
            0 if self.whole_audio else random.randint(0, frame_len - self.crop_len)
        )
        units_frame_len = frame_len if self.whole_audio else self.crop_len
        aug_flag = random.choice([True, False]) and self.use_aug

        mel = mel[start_frame : start_frame + units_frame_len]

        units = units[start_frame : start_frame + units_frame_len]

        # load f0
        aug_shift = 0
        if aug_flag:
            aug_shift = features["keyshift"]
        f0_frames = (
            2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        )

        # load volume
        volume_frames = volume[start_frame : start_frame + units_frame_len]

        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()

        # Get spk_id from first part of name_ext path
        spk_id = int(name_ext.split(os.path.sep)[0])
        spk_id = torch.LongTensor(np.array([spk_id])).to(self.device)

        return dict(
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            aug_shift=aug_shift,
            name=name,
            name_ext=name_ext,
        )

    def __len__(self):
        return len(self.paths)
