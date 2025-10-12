# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from dataset_api.dataset_registry import DatasetInterfaceRegistry

from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()
    # 🔧 Force GigaHands to use BERT + Transformer Decoder
    if args.dataset in ["gigahands", "humanml"]:  # since you remapped humanml -> GigaHands
        args.text_encoder_type = "bert"
        args.arch = "trans_dec"
        fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("📋 Parsed args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),)

    args.save_interval = 5000
    print("creating model and diffusion...")
    if not hasattr(args, 'text_encoder_type') or args.text_encoder_type is None:
        raise ValueError("[ERROR] '--text_encoder_type' is missing from args or not parsed correctly.")
    else:
        print(f"✅ text_encoder_type = {args.text_encoder_type}")

    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    if model.rot2xyz is not None:
        model.rot2xyz.smpl_model.eval()


    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
