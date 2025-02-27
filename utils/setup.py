import glob
import logging
import wandb
import os, random
from pathlib import Path
from functools import partial
from datetime import datetime


def make_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "[%Y/%m/%d %H:%M]")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def log(msg, use_wandb=False, logger=None):
    if logger:
        if use_wandb:
            wandb.log(msg)
        else:
            logger.info(msg)

def init_setup(args):
    if args.feature_path is None:
        args.feature_path = Path.cwd() / 'feature'
        args.feature_path.mkdir(exist_ok=True)
        args.feature_path = args.feature_path / f"{args.dataset_type}_{args.num_train}_feature.npy"

    if args.code_path is None:
        args.code_path = Path.cwd() / 'code'
        args.code_path.mkdir(exist_ok=True)
        args.code_path = args.code_path / f"{args.dataset_type}_{args.num_train}_hqt.npy"

    args.proj = Path.cwd() / args.proj
    args.proj.mkdir(exist_ok=True)
    if args.exp is None:
        args.exp = f'{args.dataset_type}_{args.hash_model}_{args.encode_length}bits_train_{args.num_train}_query_{args.num_query}'
        if args.hqt:
            args.exp += "_hqt"
    version_id = len(list(Path(args.proj).glob(f"{args.exp}_v*")))
    args.exp = args.proj / f"{args.exp}_v{version_id}"
    args.exp.mkdir(exist_ok=True)
    args.text_log_path = args.exp / "log.txt"
    args.best_weight_path = args.exp / "best_weight.pth"
    args.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # init logger
    args.logger = make_logger(args.text_log_path)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=str(args.exp).split('/')[-1])
    args.log = partial(log, logger=args.logger)
    args.print_m = f"{args.hash_model}({args.arch})"
    if args.hqt:
        args.print_m += "_hqt"


if __name__ == "__main__":
    from config import get_args_parser
    args = get_args_parser()
    print(get_args_parser())
    init_setup(args)