""" Train embedding networks """

import argparse
from datetime import datetime
from front_end.dataset import TrainDataset, EvalDataset, TrainBatchSampler
from front_end.model_former import Former
from front_end.model_res2net import Res2Net
from front_end.model_resnet import ResNet
from front_end.model_tdnn import TDNN
from front_end.trainer import Trainer
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.my_utils import init_logger


# Set global parameters
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--device', default='0,1', help='devices used for training')
parser.add_argument('--port', default='12355', help='port for ddp training')
parser.add_argument('--n_workers', type=int, default=4, help='No. of cpu threads per process')
parser.add_argument('--model', default='confusionformer',
                    help='tdnn, resnet, res2net, conformer, confusionformer, transformer')
parser.add_argument('--batch_size', type=int, default=128, help='local mini-batch size')
parser.add_argument('--epochs', type=int, default=30, help='No. of training epochs')
parser.add_argument('--train_src', default='vox2', help='vox2, cnceleb, sre21')
parser.add_argument('--save_ckpts', type=int, default=5, help='No. of ckpts to be saved')
parser.add_argument('--disable_eval', action='store_true', default=False, help='disable validation during training')
parser.add_argument('--grad_norm', action='store_true', default=False, help='monitor grad norm during training')

parser.add_argument('--repeats', type=int, default=0, help='No. of repeat of spks during one epoch')
parser.add_argument('--feat_dim', type=int, default=80, help='dimension of acoustic features')
parser.add_argument('--min_len', type=int, default=300, help='minimum No. of frames of a training sample')
parser.add_argument('--max_len', type=int, default=300, help='maximum No. of frames of a training sample')
parser.add_argument('--seed', type=int, default=20230708, help='train dataloader seed')
parser.add_argument('--speed_aug', action='store_true', default=False, help='apply SpeedPerturb')
parser.add_argument('--spec_aug', action='store_true', default=False, help='apply SpecAugment')
parser.add_argument('--noise_reverb_aug', action='store_true', default=True, help='apply NoiseReverbAugment')

parser.add_argument('--filters', default='1024-1024-1024-1024-1536',
                    help='No. of convolutional filters, 512-512-512-512-1536, 32-32-64-128-256')
parser.add_argument('--kernel_sizes', default='5-3-3-3-1',
                    help='kernel size of convolutional layers, 5-3-3-1-1, 5-1-3-1-3-1-3-1-1, 3-3-3-3-3')
parser.add_argument('--dilations', default='1-2-3-4-1',
                    help='dilation of convolutional layers, 1-2-3-1-1, 1-1-2-1-3-1-4-1-1, 1-1-1-1-1')
parser.add_argument('--pooling', default='ctdstats-128-0', help='stats, attention-256-1, ctdstats-128-0')
parser.add_argument('--emb_dims', default='192', help='embedding network config, 512-512')
parser.add_argument('--output_act', default='amsoftmax-0.25-30', help='softmax, amsoftmax-0.25-30, aamsoftmax-0.25-30')
parser.add_argument('--former_cfg', default='(12,256,0.15,0,1e0,0,1e-12,0)|(4,127,2,0.1)|(4,0.1)|(2,15,0.1)|'
                    '(16,16,8,5,0,0)|fusion', help='(n_blks, d_model, drop_path, layer_scale, ls_init, '
                    'rmsn, ln_eps, mfa), (n_heads, max_rel_dist, att_ds, att_drop), (ff_expansion, ff_drop), '
                    '(conv_expansion, conv_kernel, conv_drop), (stft_cfg), rel_att_type')

parser.add_argument('--optim', default='sgd', help='adam or sgd')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='L2 weight decay 1e-4, 2e-5')
parser.add_argument('--sync_bn', action='store_true', default=False, help='apply SyncBatchNorm')
parser.add_argument('--lr', default='lin_1_cos_1:0.01@0,0.1@3,0.0001@30', help='lin_1_cos_1:0.01@0,0.1@5,0.0001@40')
parser.add_argument('--ema_decay', type=float, default=0., help='decay rate of EMA of model weights')
parser.add_argument('--ckpt_dir', nargs='?', help='directory of model checkpoint')
parser.add_argument('--ckpt_num', nargs='?', type=int, help='checkpoint number for resuming training, default: None')
parser.add_argument('--save_freq', type=int, default=1, help='frequency to save the model in epochs')
parser.add_argument('--log_dir', default='log', help='log directory')
args = parser.parse_args()


def train_func(rank, n_gpus):
    # --------------------------------------------------------------------------------------------------
    # Initialize ckpt_dir and logger
    # --------------------------------------------------------------------------------------------------
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = f'model_ckpt/ckpt_{cur_time}' if args.ckpt_dir is None else args.ckpt_dir
    ckpt_time = '_'.join(ckpt_dir.split('/')[-1].split('_')[1:])
    Path(f'{ckpt_dir}').mkdir(parents=True, exist_ok=True)

    Path(f'{args.log_dir}').mkdir(parents=True, exist_ok=True)
    log_path = f'{args.log_dir}/log_{ckpt_time}.log'
    logger = init_logger(logger_name='train', log_path=log_path, device=rank, n_gpus=n_gpus)

    logger.info('----------------------------------------------------')
    logger.info(f'[*] ckpt_dir: {ckpt_dir}')  # log this first

    if args.model in ['tdnn', 'resnet', 'res2net']:
        logger.info(f'[*] filters: {args.filters}')
        logger.info(f'[*] kernel_sizes: {args.kernel_sizes}')
        logger.info(f'[*] dilations: {args.dilations}')
    elif args.model in ['conformer', 'confusionformer', 'transformer']:
        logger.info(f'[*] former_cfg: {args.former_cfg}')
    elif args.model == 'poolformer':
        logger.info(f'[*] poolformer_cfg: {args.poolformer_cfg}')

    for arg, val in vars(args).items():
        if arg not in ['ckpt_dir', 'filters', 'kernel_sizes', 'dilations', 'former_cfg', 'poolformer_cfg']:
            logger.info(f'[*] {arg}: {val}')
    logger.info('----------------------------------------------------\n')

    # --------------------------------------------------------------------------------------------------
    # Initialize DDP
    # --------------------------------------------------------------------------------------------------
    if n_gpus > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=n_gpus)

    # --------------------------------------------------------------------------------------------------
    # Create training and test dataloaders
    # --------------------------------------------------------------------------------------------------
    # train dataloader
    train_dataset = TrainDataset(
        source=args.train_src, mode='train', min_len=args.min_len, max_len=args.max_len,
        repeats=args.repeats, speed_aug=args.speed_aug, noise_reverb_aug=args.noise_reverb_aug)
    train_sampler = TrainBatchSampler(train_dataset, batch_size=args.batch_size, seed=args.seed)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.n_workers,
        collate_fn=train_dataset.segment_batch)

    # evaluation dataloader
    if not args.disable_eval:
        eval_dataloader = {'enroll': [], 'test': []}
        src = 'voxceleb1' if 'vox' in args.train_src else 'cnceleb1'
        keys = ['test'] if 'vox' in args.train_src else ['enroll', 'test']

        for k in keys:
            n_utts = EvalDataset(source=f'{src}_{k}', n_gpus=n_gpus).n_utterance

            for i in range(n_gpus):
                start_idx, end_idx = i * (n_utts // n_gpus), (i + 1) * (n_utts // n_gpus)
                dataset = EvalDataset(source=f'{src}_{k}', start=start_idx, end=end_idx, n_gpus=n_gpus)
                eval_dataloader[k].append(DataLoader(dataset=dataset, num_workers=2))
    else:
        eval_dataloader = None

    # --------------------------------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------------------------------
    model_args = {
        'feat_dim': args.feat_dim, 'filters': args.filters, 'kernel_sizes': args.kernel_sizes,
        'dilations': args.dilations, 'pooling': args.pooling, 'emb_dims': args.emb_dims,
        'n_class': train_dataset.n_speaker, 'output_act': args.output_act,
        'spec_aug': args.spec_aug, 'logger': logger}

    if args.model == 'tdnn':
        model = TDNN(**model_args)
    elif args.model == 'resnet':
        model = ResNet(**model_args)
    elif args.model == 'res2net':
        model = Res2Net(**model_args)
    elif args.model in ['conformer', 'confusionformer', 'transformer']:
        model = Former(**model_args, cfg=args.former_cfg, name=args.model)
    else:
        raise NotImplementedError

    logger.info('===============================================')
    logger.info(model)
    total_paras = sum(para.numel() for para in model.parameters() if para.requires_grad)
    enc_paras = sum(para.numel() for para in model.spk_model.spk_encoder.parameters() if para.requires_grad)
    logger.info(f'No. of total parameters: {total_paras / 1e6:.3f} M')
    logger.info(f'No. of encoder parameters: {enc_paras / 1e6:.3f} M\n')
    logger.info('===============================================\n')

    # --------------------------------------------------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------------------------------------------------
    trainer_args = {
        'train_dataloader': train_dataloader, 'test_dataloader': None, 'eval_dataloader': eval_dataloader,
        'model': model, 'optim': args.optim, 'weight_decay': args.weight_decay, 'lr': args.lr,
        'epochs': args.epochs, 'device': rank, 'sync_bn': args.sync_bn, 'ema_decay': args.ema_decay,
        'ckpt_dir': ckpt_dir, 'ckpt_num': args.ckpt_num, 'save_freq': args.save_freq,
        'save_ckpts': args.save_ckpts, 'grad_norm': args.grad_norm, 'logger': logger}

    trainer = Trainer(**trainer_args)

    return trainer.train()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    num_gpus = len(args.device.split(','))

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # torch.cuda.empty_cache()

    if num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        torch.multiprocessing.spawn(train_func, args=(num_gpus, ), nprocs=num_gpus, join=True)
    else:
        train_func(0, 1)

    print('To the END.\n\n')
