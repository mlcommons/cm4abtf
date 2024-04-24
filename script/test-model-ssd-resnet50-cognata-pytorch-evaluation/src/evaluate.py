"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import shutil
import importlib
import pprint
import csv
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, SSDLite, ResNet, MobileNetV2
from src.utils import generate_dboxes, Encoder, coco_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate, cognata_eval
from src.dataset import collate_fn, CocoDataset, Cognata, prepare_cognata, train_val_split

def cleanup():
    torch.distributed.destroy_process_group()

def prepare(dataset, params, rank, world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, sampler=sampler, **params)
    
    return dataloader

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco",
                        help="the root folder of dataset")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD")

    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
                        help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--batch-size", type=int, default=32, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")

    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--local-rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=65, help="number of total epochs model was trained for evaluation")
    parser.add_argument("--full-eval", action='store_true', help="Evaluate several checkpoints of a model")

    
    args = parser.parse_args()
    return args


def main(rank, opt, world_size):
    test_params = {"batch_size": opt.batch_size,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    config = importlib.import_module('config.' + opt.config)
    image_size = config.model['image_size']
    num_classes = len(coco_classes)
    if opt.model == "ssd":
        dboxes = generate_dboxes(config.model, model="ssd")
    else:
        dboxes = generate_dboxes(model="ssdlite")
    if opt.dataset == 'Cognata':
        folders = config.dataset['folders']
        cameras = config.dataset['cameras']
        ignore_classes = [2, 25, 31]
        if 'ignore_classes' in config.dataset:
            ignore_classes = config.dataset['ignore_classes']
        files, label_map, label_info = prepare_cognata(opt.data_path, folders, cameras, ignore_classes)
        files = train_val_split(files)
        test_set = Cognata(label_map, label_info, files['val'], ignore_classes, SSDTransformer(dboxes, image_size, val=True))
        num_classes = len(label_map.keys())
        print(label_map)
        print(label_info)
    elif opt.dataset == 'Coco':
        test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, image_size, val=True))
    if opt.model == "ssd":
        model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)
    else:
        model = SSDLite(backbone=MobileNetV2(), num_classes=len(coco_classes))
    test_loader = prepare(test_set, test_params, rank, world_size)

    encoder = Encoder(dboxes)

    if torch.cuda.is_available():
        model.cuda()

        if opt.amp:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
        # It is recommended to use DistributedDataParallel, instead of DataParallel
        # to do multi-GPU training, even if there is only a single node.
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)
    writer = SummaryWriter(opt.log_path)
    if opt.dataset == 'Cognata':
        if opt.full_eval:
            file_path= os.path.join(opt.log_path, opt.config + '_results.csv')
            
            with open(file_path, 'a+', newline='') as csvfile:
                fieldnames = [label_info[key] for key in sorted(label_info.keys())]
                fieldnames = ['epoch'] + fieldnames[1:]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for i in range(opt.checkpoint_freq, opt.epochs+1, opt.checkpoint_freq):
                    checkpoint = torch.load(opt.pretrained_model + '_ep' + str(i) + '.pth')
                    epoch = checkpoint["epoch"] + 1
                    model.module.load_state_dict(checkpoint["model_state_dict"])
                    metrics = cognata_eval(model, test_loader, epoch, writer, encoder, opt.nms_threshold)
                    writer.writerow([epoch] + metrics['map_per_class'].tolist())
        else:
            checkpoint = torch.load(opt.pretrained_model)
            epoch = checkpoint["epoch"] + 1
            model.module.load_state_dict(checkpoint["model_state_dict"])
            metrics = cognata_eval(model, test_loader, epoch, writer, encoder, opt.nms_threshold)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(metrics)
    else:
        checkpoint = torch.load(opt.pretrained_model)
        epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)
    cleanup()

if __name__ == "__main__":
    opt = get_args()
    torch.distributed.init_process_group("nccl", init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    main(rank, opt, world_size)

