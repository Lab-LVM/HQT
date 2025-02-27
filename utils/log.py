import csv
import json
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from filelock import FileLock

def csv2dict(csv_data):
    return {col: [row[c] for row in csv_data[1:]] for c, col in enumerate(csv_data[0])}

def write_csv(result_path, data, mode='w'):
    with open(result_path, mode, newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

class Result:
    def __init__(self, log_path='log'):
        self.result_path = os.path.join(log_path, 'result.csv')
        self.headers = ['no', 'start_time', 'duration', 'dataset_type', 'num_train', 'num_query', 'hqt', 'hash_model', 'epochs',
                        'best_map', 'encode_length', 'exp']
        self.setup_directory()
        self.setup_logfile()

    def setup_directory(self):
        Path(os.path.dirname(self.result_path)).mkdir(exist_ok=True, parents=True)

    def setup_logfile(self):
        if not Path(self.result_path).exists():
            write_csv(self.result_path, [self.headers], mode='w')

    def read_result(self):
        csv_data = []
        with open(self.result_path, 'r') as f:
            for line in csv.reader(f):
                csv_data.append(line)
        return csv_data, csv2dict(csv_data)

    def arg2result(self, args, metric):
        result = [self.get_no()]
        for column_name in self.headers[1:]:
            if hasattr(args, column_name):
                result.append(getattr(args, column_name))
            elif metric.get(column_name, None):
                result.append(metric.get(column_name, None))
            else:
                result.append('')
                print("Args and Metric object does not have : {}".format(column_name))
        return result

    def get_no(self):
        csv_list, csv_dict = self.read_result()
        return len(csv_list)

    def summary(self, args, metric):
        space = 16
        num_metric = 4
        duration, best_epoch, best_map, hash_m = list(metric.values())
        args.log('-' * space * num_metric)
        args.log(("{:>16}" * num_metric).format('Duration', 'BestEpoch', 'Best@Map', 'Hash_model'))
        args.log('-' * space * num_metric)
        args.log(
            f"{duration:>{space}}{best_epoch:{space}.5f}{best_map:{space}.5f}{hash_m:>{space}}")
        args.log('-' * space * num_metric)

        with FileLock("{}.lock".format(self.result_path)):
            result = self.arg2result(args, metric)
            write_csv(self.result_path, [result], mode='a')

    def dump_args(self, args):
        with open(os.path.join(str(args.exp), 'cmd.json'), 'wt') as f:
            keys_to_remove = ('device', 'logger', 'log')
            cmd = dict({key: str(val) for key, val in args.__dict__.items() if key not in keys_to_remove})
            json.dump(cmd, f, indent=4, ensure_ascii=False)

    def dump_metric(self, log_dir, **kwargs):
        columns = list(kwargs.keys())
        df = pd.DataFrame(data=kwargs, columns=columns, index=None).astype(float)
        df.to_csv(os.path.join(log_dir, 'metric.csv'), index=False)

    def save_result(self, args, map_list, metric):
        self.summary(args, metric)
        self.dump_args(args)
        self.dump_metric(args.exp, map=map_list)

def save_checkpoint(save_dir, model, optimizer, epoch, queryB, queryL, retrievalB, retrievalL, is_best=False):
    pairs = [('state_dict', model), ('optimizer', optimizer)]
    checkpoint_dict = {k: v.state_dict() for k, v in pairs if v}
    checkpoint_dict['epoch'] = epoch
    if is_best:
        torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_best.pth'))
        np.save(os.path.join(save_dir, "query_binary.npy"), queryB.cpu().numpy())
        np.save(os.path.join(save_dir, "query_label.npy"), queryL.cpu().numpy())
        np.save(os.path.join(save_dir, "retrieval_binary.npy"), retrievalB.cpu().numpy())
        np.save(os.path.join(save_dir, "retrieval_label.npy"), retrievalL.cpu().numpy())
