"""Training script
This is the training script for superpoint detector and descriptor.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import sys
sys.path.insert(0, "../ProxyOpt/")
sys.path.insert(0, "../fast-openISP/")
sys.path.insert(0, "../ProxyOpt/pytorch-msssim/")
from ISP_tools.ProxyISPDataset import ProxyISPDataset, EXPERIMENT_OUTPUT_PATH
from proxy_utils import extract_iteration
from model import U_Net
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter

import argparse
import yaml
import os
import logging

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

# from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
# from models.model_wrap import SuperPointFrontend_torch, PointTracker

###### util functions ######
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass

from utils.loader import get_save_path

###### util functions end ######


###### train script ######
def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)
    pass

# def train_joint_dsac():
#     pass

def train_joint(config, output_dir, args):
    torch.multiprocessing.set_start_method('spawn')
    
    train_config_path = config["proxyopt"]["config_path"]
    proxy, proxy_isp_dataset = load_proxy_model_and_dataset(train_config_path)

    proxy_writer = SummaryWriter(f"logs/{args.exper_name}/logs")

    assert 'train_iter' in config

    # config
    # from utils.utils import pltImshow
    # from utils.utils import saveImg
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    # writer = SummaryWriter(getWriterPath(task=args.command, date=True))
    writer = SummaryWriter(getWriterPath(task=args.command, 
        exper_name=args.exper_name, date=True))
    ## save data
    save_path = get_save_path(output_dir)

    # data loading
    # data = dataLoader(config, dataset='syn', warp_input=True)
    data = dataLoader(config, proxy, proxy_isp_dataset, dataset=task, warp_input=True, load_dataloader = False)
    # train_loader, val_loader = data['train_loader'], data['val_loader']
    train_set, val_set = data["train_set"], data["val_set"]

    # datasize(train_loader, config, tag='train')
    # datasize(val_loader, config, tag='val')
    # init the training agent using config file
    # from train_model_frontend import Train_model_frontend
    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])

    train_agent = train_model_frontend(config, save_path=save_path, device=device)

    # writer from tensorboard
    train_agent.writer = writer
    train_agent.proxy_writer = proxy_writer

    # feed the data into the agent
    # train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader
    train_agent.train_set = train_set
    train_agent.val_set = val_set

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    # freeze sp model - BOAT
    for parameter in train_agent.net.parameters():
        parameter.requires_grad = False
    
    # remove dataParallel - BOAT
    # if dataParallel, TODO: plse go fix dataParallel to optimize proxy instead of superpoint - BOAT
    # train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        train_agent.saveModel()
        pass

def load_proxy_model_and_dataset(train_config_path):
    PROXYOPT_BASE_PATH = Path("../ProxyOpt/")
    with open(train_config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    config = yaml_dict["config"]
    openisp_config = yaml_dict["openisp_config"]
    hyp_setting = yaml_dict["hyp_setting"]

    stage2_output_dir = Path("proxyopt_output") / (config["experiment_name"] + "s1")

    if not os.path.exists(stage2_output_dir):
        os.makedirs(stage2_output_dir)
        os.makedirs(stage2_output_dir / "logs")
        os.makedirs(stage2_output_dir / "checkpoints")

    latest_stage2_obj = None
    checkpoints = list(os.scandir(stage2_output_dir / "checkpoints"))
    if checkpoints.__len__() > 0:
        latest_stage2_checkpoint_path = max(checkpoints, key = lambda x: int(x.name.split("_")[-1].split("it")[0]))
        latest_stage2_obj = torch.load(latest_stage2_checkpoint_path.path)

    output_dir = PROXYOPT_BASE_PATH / EXPERIMENT_OUTPUT_PATH / config["experiment_name"]
    checkpoint_dir = output_dir / "checkpoints"
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = os.listdir(output_dir / "checkpoints")

    latest_file = max(checkpoint_files, key=lambda f: extract_iteration(f))

    latest_obj = torch.load(output_dir / "checkpoints" / latest_file)

    config = latest_obj["config"]

    # Model and dataset initialization
    in_channels = 1
    if config["input_type"] == "stacked":
        in_channels = 4  # GRGB

    additional_conf = {
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240115_123915.dng"],
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240117_182706.dng"],
        "proxyopt_base_path": "/home/boat/proxyISP/ProxyOpt/"
    }

    dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf)

    raw, _, sample_hyp = dataset.__getitem__(0)
    param_number = sample_hyp.shape[-1]

    net = U_Net(in_channels, 3, step_flag=3, img_size=config["img_size"], param_number=param_number)
    net.load_state_dict(latest_obj["model_state_dict"])
    net = net.to("cuda")

    # Setup target and starting hyperparameters
    dataset.switch_stage2()
    net.img_size = dataset.target_size
    start_hyp = dataset.get_original_hyp(True, True, add_eps = False)
    net.load_param_layer(start_hyp)

    if latest_stage2_obj is not None:
        net.param_layer = latest_stage2_obj["param_layer"]
        net.param_layer.requires_grad = True

    net.set_requires_param_layer_grad(True)

    return net, dataset

if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_base)

    # Training command
    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_joint)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)


