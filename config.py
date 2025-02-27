import os, random
import numpy
import torch
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Unsupervised Hashing with Hyper Quantization Tree")
    parser.add_argument('--data-path', type=str, default='../../datasets', help='Path to the dataset')
    parser.add_argument('--feature-path', type=str, default=None, help='Path to the extracted feature')
    parser.add_argument('--code-path', type=str, default=None, help='Path to the hashing code')
    parser.add_argument('--proj', type=str, default='exp', help='Project name')
    parser.add_argument('--exp', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=str, default='./hqt_bihalf_cifar10_I_16.pth', help='Checkpoint path')
    # data
    data = parser.add_argument_group(description='Dataset parameters')
    data.add_argument('--dataset-type', type=str, default='cifar10', help='Dataset type')
    data.add_argument('--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+', help='Image mean')
    data.add_argument('--std', type=float, default=(0.229, 0.224, 0.225), nargs='+', help='Image std')
    data.add_argument('--scale', type=float, default=(0.2, 1.), nargs='+', help='Scale of RandomResizeCrop')
    data.add_argument('--num-query', type=int, default=10000, help="Number of query data.")
    data.add_argument('--num-train', type=int, default=5000, help="Number of training data.")
    data.add_argument('--img-size', type=int, default=224, help='Resize of image size')
    data.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size')
    data.add_argument('-w', '--workers', type=int, default=4, help='Number of workers')
    # train
    train = parser.add_argument_group('Train parameters')
    train.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    train.add_argument('--lr-decay', type=int, default=120, help='Learning rate epoch decrease')
    train.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    train.add_argument('-o', '--optimizer', type=str, default='sgd', help='Optimizer (default: "sgd")')
    train.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum of optimizer')
    train.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    # model
    model = parser.add_argument_group('Model parameters')
    model.add_argument('-a', '--arch', type=str, default='vgg16', help='backbone architecture')
    model.add_argument('--hash-model', type=str, default='bihalf', help='Name of the hashing model')
    model.add_argument('--use-timm', action='store_true', default=False, help='Use timm model')
    # hashing
    hashing = parser.add_argument_group(description='Hash parameters')
    hashing.add_argument('-c', '--encode-length', type=int, default=16, help='Binary hash code length. (default:16)')
    hashing.add_argument('--gamma', type=int, default=6, help='BihalfNet parameter gamma. (default:6)')
    hashing.add_argument('--alpha', type=float, default=0.5, help='HQT parameter alpha. (default:0.5)')
    hashing.add_argument('--hqt', action='store_true', default=False, help='Use HQT (default:False)')
    hashing.add_argument('--min-depth', type=int, default=3, help="HQT minimum depth (default:3)")
    hashing.add_argument('--max-depth', type=int, default=5, help="HQT maximum depth (default:5)")
    # setup
    setup = parser.add_argument_group(description='Setup parameters')
    setup.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    setup.add_argument('-g', '--gpu', type=int, default=0, help='Using gpu.(default: False)')
    setup.add_argument('--use-wandb', action='store_true', help='Use wandb.')
    setup.add_argument('--test-map', type=int, default=20, help='Setting number of test map')
    setup.add_argument('-k', '--topk', type=int, default=-1, help='Calculate map of top k.(-1: map@all)')
    return parser

def get_args_parser():
    args = args_parser().parse_args()
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda")
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        torch.cuda.set_device(args.gpu)
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args

