import sys
import os

import json
import argparse
import torch
import random
import numpy as np

from model.FFTRadNet import FFTRadNet
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader_mod import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation_dct_0th_order_optimize import run_FullEvaluation_SGD
from utils.metrics import GetFullMetrics, Metrics
import torch.nn as nn

import pickle
from scipy.stats import hmean

def main(config, checkpoint, difficult, args):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)
    
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=difficult)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])


    # Create the model
    net = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                        channels = config['model']['channels'], 
                        regression_layer = 2, 
                        detection_head = config['model']['DetectionHead'], 
                        segmentation_head = config['model']['SegmentationHead'])

    net.to('cuda')


    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint,weights_only=False) # PyTorch 2.6
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    print('lambda_val:', args.lambda_val)

    result_dict = run_FullEvaluation_SGD(net,test_loader,enc,args,config,
                                        quantize=True,
                                        result_only=args.result_only)
    if args.result_only:
        cr = np.asarray(result_dict['dct_info']['cr_p'])
    else:
        predictions, dct_info, perfs_all = result_dict
        cr = np.asarray(dct_info['cr_p'])
        result_dict = {'predictions': predictions, 
                    'dct_info': dct_info, 
                    'perfs_all':perfs_all}
        
    print(f"Compression ratio = {hmean(cr)}")
    result_dict['checkpoint'] = checkpoint
    result_dict['compression_ratio'] = cr

    os.makedirs(args.result_dir, exist_ok=True)

    tokens = []
    if args.loss_type == "balance":
        tokens.append("balance")
    if args.init_cr_per_scene:
        tokens.append("init")

    tokens += [
        f"obj_{args.objective}"        if args.loss_type == "balance" else None,
        f"qbit_{args.qbit}",
        f"cr_{args.comp_ratio}",
        f"lr_{args.lr}",
        f"eps_{args.epsilon}",
        f"ct_{args.conf_thd}",
        f"lam_{args.lambda_val}",
        f"gc_{args.grad_clip}",
    ]
    filename = "_".join(t for t in tokens if t is not None) + ".pkl"
    save_dir = os.path.join(args.result_dir, filename)

    with open(save_dir, 'wb') as f:
        pickle.dump(result_dict, f)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    # compression related
    parser.add_argument('-cr', '--comp_ratio', default=1, type=float,
                        help='compression ratio')
    parser.add_argument('--qbit', default=8, type=int,
                        help='quantization bit width')
    parser.add_argument('--BL', default=64, type=int,
                        help='block length')
    # feedback related
    parser.add_argument('--enable_feedback', default=False, action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='perturbation amount')
    parser.add_argument('--min_comp_ratio', default=1, type=float,
                        help='min compression ratio')
    parser.add_argument('--max_comp_ratio', default=100, type=float,
                        help='max compression ratio')
    parser.add_argument('--conf_thd', default=0.0, type=float,
                        help='filter threshold for confidence')
    parser.add_argument('--loss_type', default=None, type=str,
                        help='filter threshold for confidence')
    parser.add_argument('--lambda_val', default=0.0, type=float,
                        help='filter threshold for confidence')
    parser.add_argument('--grad_clip', default=10.0, type=float,
                        help='Clip gradient if it is larger than a threshold')
    parser.add_argument('--objective', default='add',type=str,
                        help='Objective function')
    parser.add_argument('--init_cr_per_scene', default=False, action='store_true',
                        help='initialize comp ratio per scene')
    # OOD scenario
    parser.add_argument('--OOD', default=False, action='store_true')
    parser.add_argument('--ood_type', default='rect', type=str,
                        help='OOD type: rect, ...')
    parser.add_argument('--snr', default=1, type=float,
                        help='snr for OOD noise injection')
    parser.add_argument('--period', default=0, type=int,
                        help='window period for OOD injection if 0 always on')
    
    parser.add_argument('--result_dir', default='results', type=str,
                        help='result directory')
    parser.add_argument('--result_only', default=False, action='store_true',
                        help='save final results only (if unflagged, save frame-wise results)')
    
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint, args.difficult, args)

