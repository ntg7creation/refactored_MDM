# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import logging
from utils.fixseed import fixseed

# from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_test import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform
from dataset_api.dataset_registry import DatasetInterfaceRegistry

from types import SimpleNamespace

def main():
    import argparse

    # Minimal parser to get dataset name
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    cli_args = parser.parse_args()
    dataset_name = cli_args.dataset
    # args = train_args()
    # args_dict = vars(args)
    # # Log to console
    # print("📋 Parsed CLI + default args from train_args():")
    # for k, v in args_dict.items():
    #     print(f"  {k}: {v}")
    # fixseed(args.seed)

    # Save dataset name early for use in logging, reporting, naming
    # dataset_name = args.dataset
    print(f"🧩 Attempting to load dataset interface: {dataset_name}")
    dataset_interface = DatasetInterfaceRegistry.get(dataset_name)


    # Load full YAML config into my_args
    config = dataset_interface.get_config()
    args = SimpleNamespace(**config)

    # Manually inject runtime-resolved values (not stored in YAML)
    args.dataset = dataset_name
    args.save_dir = dataset_interface.get_config_value("save_dir")  # Already extracted elsewhere too
    args.device = dataset_interface.get_config_value("device")

    fixseed(args.seed)

    # ---------------------------- Save dir and logging ---------------------------- #
    # logging.info(f"🧾 save_dir from args before override: {args.save_dir}")
    save_dir = dataset_interface.get_config_value("save_dir")
    if save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 Created save directory: {save_dir}")
    else:
        print(f"📂 Save directory already exists: {save_dir}")

    # ✅ Set up logging to file and console
    log_file_path = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console_handler)

    # 📝 Log starting info
    logging.info(f"🧩 Loaded dataset interface: {dataset_name}")
    logging.info(f"📁 save_dir from config: {save_dir}")

    # Save raw CLI args for reference
    args_path = os.path.join(save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    logging.info(f"✅ Raw CLI args saved to {args_path}")

    # ---------------------------------- Init Platform ---------------------------------- #
    logging.info("🚀 Starting training run")

    # logging.info(f"📦 train_platform_type from args before override: {args.train_platform_type}")
    train_platform_type_str = dataset_interface.get_config_value("train_platform_type")
    logging.info(f"📥 train_platform_type from config: {train_platform_type_str}")
    train_platform_type = eval(train_platform_type_str)
    # train_platform_type = eval(args.train_platform_type)

    train_platform = train_platform_type(save_dir)
    # train_platform = train_platform_type(args.save_dir)

    train_platform.report_args(dataset_interface.get_config(), name='Args')

    # ---------------------------- Device Setup ---------------------------- #
    # logging.info(f"🧾 device from args before override: {args.device}")
    device = dataset_interface.get_config_value("device")
    logging.info(f"🖥️ device from config: {device}")
    dist_util.setup_dist(device)
    # dist_util.setup_dist(args.device)

    # ---------------------------- Data Loader Setup ---------------------------- #
    # logging.info(f"🧾 batch_size from args before override: {args.batch_size}")
    batch_size = dataset_interface.get_config_value("batch_size")
    logging.info(f"📦 batch_size from config: {batch_size}")

    # logging.info(f"🧾 num_frames from args before override: {args.num_frames}")
    num_frames = dataset_interface.get_config_value("num_frames")
    logging.info(f"📦 num_frames from config: {num_frames}")

    # logging.info(f"🧾 pred_len from args before override: {args.pred_len}")
    pred_len = dataset_interface.get_config_value("pred_len")
    logging.info(f"📦 pred_len from config: {pred_len}")

    # logging.info(f"🧾 context_len from args before override: {args.context_len}")
    context_len = dataset_interface.get_config_value("context_len")
    logging.info(f"📦 context_len from config: {context_len}")

    # logging.info(f"🧾 save_interval from args before override: {args.save_interval}")
    save_interval = dataset_interface.get_config_value("save_interval")
    logging.info(f"📦 save_interval from config: {save_interval}")

    logging.info("📦 Creating data loader...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=batch_size,
        num_frames=num_frames,
        fixed_len=pred_len + context_len,
        pred_len=pred_len,
        device=device,
    )

    # ---------------------------- Model Setup ---------------------------- #
    logging.info("🧠 Creating model and diffusion...")

    # logging.info(f"🧾 text_encoder_type from args before override: {args.text_encoder_type}")
    text_encoder_type = dataset_interface.get_config_value("text_encoder_type")
    if text_encoder_type is None:
        raise ValueError("[ERROR] 'text_encoder_type' is missing from the config file.")
    logging.info(f"✅ text_encoder_type from config: {text_encoder_type}")


    config = dataset_interface.get_config()

  

    config = dataset_interface.get_config()

    # Merge CLI args and config, CLI overrides config if value is not None
    merged_args = config.copy()
    for k, v in vars(args).items():
        if v is not None:
            merged_args[k] = v

    my_args = SimpleNamespace(**merged_args)

    # Inject values not in either config or CLI
    my_args.device = device
    my_args.save_dir = save_dir
    my_args.dataset = dataset_name

    # # 🧪 Compare args (CLI) vs. my_args (merged config)
    # logging.info("🔍 Comparing CLI args vs merged config (my_args):")

    # for key in sorted(vars(my_args)):
    #     cli_val = getattr(args, key, None)
    #     merged_val = getattr(my_args, key, None)

    #     if cli_val != merged_val:
    #         logging.info(f"🔁 {key}: CLI = {cli_val!r} | Used = {merged_val!r}")
    #     else:
    #         logging.info(f"✅ {key}: {merged_val!r}")

    # print("🔍 my_args:")
    # for k, v in vars(my_args).items():
    #     print(f"  {k}: {v}")

    model, diffusion = create_model_and_diffusion(my_args, data)

    assert hasattr(model, "cond_mask_prob")
    print(f"[CFG] cond_mask_prob in model: {model.cond_mask_prob}")
    assert abs(model.cond_mask_prob - args.cond_mask_prob) < 1e-8

    model.to(dist_util.dev())
    if model.rot2xyz is not None:
        model.rot2xyz.smpl_model.eval()

    total_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    logging.info(f"📊 Total params: {total_params:.2f}M")

    # ---------------------------- Training Loop ---------------------------- #
    logging.info("🏁 Starting training loop...")
    TrainLoop(args, train_platform, model, diffusion, data,dataset_interface).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
