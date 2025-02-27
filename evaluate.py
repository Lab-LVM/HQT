import os
import numpy as np
import torch
from models.utils import get_model
from data.utils import get_dataset

def evaluate(model, query_dataloader, retrieval_dataloader, device, hash_model='bihalf', topk=-1):
    queryB, queryL = compress(model, query_dataloader, hash_model,device)
    retrievalB, retrievalL = compress(model, retrieval_dataloader, hash_model,device)
    mAP = mean_average_precision(
        queryB,
        retrievalB,
        queryL,
        retrievalL,
        device,
        topk,
    )
    model.train()

    return mAP, queryB, queryL, retrievalB, retrievalL


def compress(model, dataloader, hash_model, device):
    bs, classes = [], []
    model.eval()
    for img, img_aug, cls, index in dataloader:
        img = img.cuda()
        if hash_model == 'bihalf':
            _, outputs, _ = model(img, img_aug)
            code = torch.sign(outputs)
        classes.append(cls)
        bs.append(code.data)
    B = torch.cat(bs)
    L = torch.cat(classes)
    # if dataset_type.lower().startswith('cifar'):
    #     L = F.one_hot(L.to(torch.int64), 10)
    return B.to(device), L.to(device)

def generate_code(model, dataloader, hash_model, code_length, device):
    """
    https://github.com/luoxiao12/CIMON/blob/main/cimon.py
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            if hash_model == 'bihalf':
                _, outputs, _ = model(data, _)
                code[index, :] = outputs.sign().cpu()

    return code

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    https://github.com/luoxiao12/CIMON/blob/main/evaluate.py
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance

        if topk == -1:
            retrieval = retrieval[torch.argsort(hamming_dist)]
        else:
            retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return float(mean_AP)


def validation(args):
    model = get_model(args.hash_model, encode_length=args.encode_length, arch=args.arch, use_timm=args.use_timm).to(
        args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    _, query_loader, retrieval_loader = get_dataset(args.dataset_type,
                                                    root=args.data_path,
                                                    num_query=args.num_query,
                                                    num_train=args.num_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    hash_model=args.hash_model,
                                                    mean=args.mean,
                                                    std=args.std,
                                                    img_size=args.img_size,
                                                    scale=args.scale, )
    mAP, qB, qL, rB, rL = evaluate(model, query_loader, retrieval_loader, args.device, args.hash_model, args.topk)
    print("Best@mAP:",mAP)

if __name__ == '__main__':
    from config import get_args_parser
    args = get_args_parser()
    validation(args)

