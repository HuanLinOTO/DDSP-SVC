# params
from concurrent.futures import ThreadPoolExecutor
import os
from random import random
import torch
import yaml


pretrain_model_dict = {
    "exp1": r"D:\Projects\SVCFusion\pretrained\ddsp6.1\contentvec768l12.0\model_0.pt",
    "exp2": r"D:\Projects\SVCFusion\pretrained\ddsp6.1\contentvec768l12.0\model_0.pt",
}
stop_step = 10
preview_reference_audio_src = "./path/to/it"
training_devices = [
    "cuda:0",
    # "cuda:1",
]
basic_config_path = "./configs/reflow.yaml"
# end_params


def train_single(
    name: str,
    pretrain_model: str,
    device: str,
):
    print()
    print("----------------------------------------")
    print()
    print("Training: ", name)
    print("Using device: ", device)
    print("Using pretrain model: ", pretrain_model)
    # read config
    with open(basic_config_path, "r") as config:
        args = yaml.safe_load(config)
        args["train"]["pretrain_ckpt"] = pretrain_model
        args["train"]["stop_step"] = stop_step
        args["device"] = device
        args["env"]["expdir"] = f"exp/{name}"
        args["env"]["gpu_id"] = torch.device(device).index
    # 写入临时 config
    rand_id = str(random())[2:8]
    path_config = f"./configs/tmp/{rand_id}_config.yaml"
    os.makedirs(os.path.dirname(path_config), exist_ok=True)
    with open(path_config, "w") as f:
        yaml.dump(args, f)
    os.system(f"python train_reflow.py -c {path_config}")


if __name__ == "__main__":
    device_index = 0
    futures = []
    with ThreadPoolExecutor(max_workers=len(training_devices)) as executor:
        for name in pretrain_model_dict:
            pretrain_model = pretrain_model_dict[name]
            if device_index >= len(training_devices) - 1:
                device_index = 0

            device = training_devices[device_index]
            device_index += 1
            futures.append(executor.submit(train_single, name, pretrain_model, device))
