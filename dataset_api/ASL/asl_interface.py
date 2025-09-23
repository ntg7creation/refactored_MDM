# datasets/asl_interface.py
import os
import yaml
from torch.utils.data import DataLoader
from dataset_api.dataset_interface import DatasetInterface, DatasetFunctions
from dataset_api.ASL.ASL_loader import ASLHandsML3D  # Update this import path if needed

class ASLInterface(DatasetInterface):
    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(cur_dir, "asl.yaml")  # You’ll need to create this
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_function(self, name: str):
        fn = DatasetFunctions(name)

        if fn == DatasetFunctions.GET_LOADER:
            def get_loader(args):
                dataset = ASLHandsML3D(
                    mode='train',
                    split='train',
                    device=args.device,
                    fixed_len=args.pred_len + args.context_len,
                    # root_dir=args.root_dir,
                    # annotation_file=args.annotation_file,
                    # mean_std_dir=args.mean_std_dir,
                    # side=args.side,
                    # dmvb_layout=args.dmvb_layout,
                )
                return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            return get_loader

        elif fn == DatasetFunctions.EVALUATE:
            from eval import eval_gigahands  # ⚠️ You might want a separate `eval_asl.py` if needed
            return eval_gigahands.evaluate

        elif fn == DatasetFunctions.COLLATE:
            def default_collate(batch):
                return tuple(zip(*batch))
            return default_collate

        elif fn == DatasetFunctions.WRAPPER:
            return lambda device: None  # ASL doesn’t need evaluator wrapper (if same as GigaHands)

        elif fn == DatasetFunctions.CONFIG:
            return self.config

        raise NotImplementedError(f"ASLInterface: function {name} not implemented")

    def get_config(self) -> dict:
        return self.config
