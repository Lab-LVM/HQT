import datetime
import time
from config import get_args_parser
from utils.setup import init_setup
from utils.log import save_checkpoint, Result
from models.utils import get_model, get_loss, get_optimizer, adjust_learning_rate
from data.utils import get_dataset
from HQT import HQT, getfeature
from evaluate import evaluate

def run(args):
    init_setup(args)
    p_label=None
    if args.hqt:
        feature = getfeature(args)
        p_label = HQT(feature, args.code_path, args.encode_length, args.min_depth, args.max_depth, args.seed)
    train_loader, query_loader, retrieval_loader = get_dataset(args.dataset_type,
                                                             root=args.data_path,
                                                             num_query=args.num_query,
                                                             num_train=args.num_train,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.workers,
                                                             hash_model='bihalf',
                                                             mean=args.mean,
                                                             std=args.std,
                                                             img_size=args.img_size,
                                                             scale=args.scale,
                                                             hqt_label=p_label)
    model = get_model(args.hash_model, encode_length=args.encode_length, arch=args.arch, use_timm=args.use_timm).to(args.device)
    criterion = get_loss(args.hash_model, hqt=args.hqt, alpha=args.alpha)
    optimizer = get_optimizer(model, args.hash_model, args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_map = 0
    best_epoch = 0
    count = 0
    map_list = []
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_time = time.time()
        model.train()
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay)
        for i, (img, img_aug, lbs, index) in enumerate(train_loader):
            img = img.to(args.device)
            img_aug = img_aug.to(args.device)
            lbs = lbs.to(args.device)

            optimizer.zero_grad()
            x, _, b = model(img, img_aug)
            loss = criterion(x, b, lbs)
            loss.backward()
            optimizer.step()

        space = 16
        num_metric = 3
        args.log('-' * space * num_metric)
        args.log(("{:>16}" * num_metric).format(' ', 'Time', 'Loss'))
        epoch_duration = str(datetime.timedelta(seconds=time.time() - epoch_time)).split(".")[0]
        args.log('-' * space * num_metric)
        args.log(f"{'Epoch: ' + str(epoch):>{space}}{epoch_duration:>{space}}{loss:{space}.5}")
        args.log('-' * space * num_metric)

        if (epoch + 1) % args.test_map == 0:
            count += 1
            valid_time = time.time()
            mAP, qB, qL, rB, rL = evaluate(model, query_loader, retrieval_loader, args.device, args.hash_model, args.topk)
            space = 16
            num_metric = 5
            args.log('-' * space * num_metric)
            args.log(("{:>16}" * num_metric).format(' ', 'Epoch', 'Time', 'Loss', 'mAP'))
            epoch_duration = str(datetime.timedelta(seconds=time.time() - valid_time)).split(".")[0]
            args.log('-' * space * num_metric)
            args.log(
                f"{'Validation: ' + str(count):>{space}}{str(epoch):>{space}}{epoch_duration:>{space}}{loss:{space}.5}{mAP:{space}.5}")
            args.log('-' * space * num_metric)

            if best_map < mAP:
                best_map = mAP
                best_epoch = epoch
            map_list.append(mAP)
            save_checkpoint(args.exp, model, optimizer, epoch, qB, qL, rB, rL, is_best=best_epoch == epoch)

            if args.use_wandb:
                args.log({'validation_loss': loss, 'mAP': mAP}, metric=True)

    best_map = round(float(best_map), 8)
    duration = str(datetime.timedelta(seconds=time.time() - start_time)).split('.')[0]
    Result(args.proj).save_result(args, map_list, dict(duration=duration, best_epoch=int(best_epoch), best_map=best_map,
                                                       hash_m = args.print_m))

if __name__ == "__main__":
    args = get_args_parser()
    run(args)