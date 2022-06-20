import argparse
from email.policy import default
import sys

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import Callbacks, SaverRestore
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tqdm import tqdm

from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer
from model_zoo import minkunet, spvcnn, spvnas_specialized

import numpy as np
import os

train_label_name_mapping = {
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
        4: 'other-vehicle',
        5: 'person',
        6: 'bicyclist',
        7: 'motorcyclist',
        8: 'road',
        9: 'parking',
        10: 'sidewalk',
        11: 'other-ground',
        12: 'building',
        13: 'fence',
        14: 'vegetation',
        15: 'trunk',
        16: 'terrain',
        17: 'pole',
        18: 'traffic-sign'
    }

name_label_mapping = {
        'unlabeled': 0,
        'outlier': 1,
        'car': 10,
        'bicycle': 11,
        'bus': 13,
        'motorcycle': 15,
        'on-rails': 16,
        'truck': 18,
        'other-vehicle': 20,
        'person': 30,
        'bicyclist': 31,
        'motorcyclist': 32,
        'road': 40,
        'parking': 44,
        'sidewalk': 48,
        'other-ground': 49,
        'building': 50,
        'fence': 51,
        'other-structure': 52,
        'lane-marking': 60,
        'vegetation': 70,
        'trunk': 71,
        'terrain': 72,
        'pole': 80,
        'traffic-sign': 81,
        'other-object': 99,
        'moving-car': 252,
        'moving-bicyclist': 253,
        'moving-person': 254,
        'moving-motorcyclist': 255,
        'moving-on-rails': 256,
        'moving-bus': 257,
        'moving-truck': 258,
        'moving-other-vehicle': 259
    }

def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--data-dir', type=str, help='Location of data')
    parser.add_argument('--save-dir', type=str, help='Location to save labels')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset(args.data_dir)
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    if 'spvnas' in args.name.lower():
        model = spvnas_specialized(args.name)
    elif 'spvcnn' in args.name.lower():
        model = spvcnn(args.name)
    elif 'mink' in args.name.lower():
        model = minkunet(args.name)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)
    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=configs.train.seed)
    callbacks = Callbacks([
        SaverRestore(),
        MeanIoU(configs.data.num_classes, configs.data.ignore_label)
    ])
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']

    trainer.before_train()
    trainer.before_epoch()

    model.eval()

    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        outputs = model(inputs)

        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        for idx in range(invs.C[:, -1].max() + 1):
            cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
            
            # print()
            # print(idx)
            # print(invs.C[:, -1].max())
            # print(outputs_mapped)
            # print(outputs_mapped.cpu().numpy())
            remaped = [name_label_mapping[train_label_name_mapping[lbl]] for lbl in outputs_mapped.cpu().numpy()]
            
            # print(remaped)
            # print(np.shape(outputs_mapped.cpu().numpy()))
            
            fileName = feed_dict["file_name"]
            # print(fileName)
            fileName = os.path.basename(fileName[0]).replace('.bin', '')

            inv_labels = np.array(remaped).astype('uint32')
            outputPath = args.save_dir + "/" + fileName + '.label'
            inv_labels.tofile(outputPath)
            print("Save " + outputPath)


        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)
        output_dict = {'outputs': outputs, 'targets': targets}
        trainer.after_step(output_dict)
        # print("HERE", feed_dict.items())


    trainer.after_epoch()


if __name__ == '__main__':
    main()
