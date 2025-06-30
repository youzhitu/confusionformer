""" Base trainer """

from back_end.preprocess import length_norm
from back_end.cosine_score import CosineScorer
from copy import deepcopy
from front_end.model_misc import dist_concat_all_gather
import math
import numpy as np
import os
from pathlib import Path
from time import perf_counter
import torch
import torch.distributed as dist
from utils.eval_metrics import eval_performance


class Trainer(object):
    def __init__(self, train_dataloader=None, test_dataloader=None, eval_dataloader=None, model=None, optim='sgd',
                 weight_decay=1e-4, lr='lin_1_cos_1:0.01@0,0.1@3,0.0001@30', epochs=100, device=0, sync_bn=False,
                 ema_decay=0.99992, ckpt_dir='model_ckpt', ckpt_num=None, save_freq=5, save_ckpts=5, grad_norm=False,
                 logger=None):

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.sync_bn = sync_bn
        self.ema_decay = ema_decay
        self.ckpt_dir = ckpt_dir
        self.ckpt_num = ckpt_num
        self.save_freq = save_freq
        self.save_ckpts = save_ckpts
        self.grad_norm = grad_norm
        self.logger = logger

        # Initialize training
        self.optimizer = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs_trained = 0
        self.lr_scheduler = LRScheduler(self.lr)
        grad_norm_dir = f"grad_norm/{'_'.join(ckpt_dir.split('/')[-1].split('_')[1:])}" if grad_norm else None
        self.train_metrics = Metrics(grad_norm_dir)
        self.test_metrics = Metrics() if self.test_dataloader is not None else None
        self.model_ema = None

        self.setup_train()

        # Initialize evaluation
        if self.eval_dataloader is not None:
            from utils.my_utils import rd_data_frame

            if 'vox' in self.train_dataloader.dataset.source:
                self.enroll_ids = rd_data_frame('meta/eval/voxceleb1_test_path2info',
                                                ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.test_ids = self.enroll_ids.copy()
                self.trials_file = 'trials/voxceleb1/trials_voxceleb.npz'
            else:
                self.enroll_ids = rd_data_frame('meta/eval/cnceleb1_enroll_path2info',
                                                ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.test_ids = rd_data_frame('meta/eval/cnceleb1_test_path2info',
                                              ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.trials_file = 'trials/cnceleb1/trials.npz'

    def setup_train(self):
        self.setup_model()
        self.setup_optimizer()

        if os.listdir(self.ckpt_dir):
            self.load_checkpoint(map_location=torch.device(self.device))  # Load ckpt for resuming training

        if self.ema_decay != 0.:
            self.model_ema = deepcopy(self.model).eval()

    def setup_model(self):
        self.model = self.model.to(self.device)

        if dist.is_initialized():
            if self.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

    def setup_optimizer(self):
        if self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        elif self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif self.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        else:
            raise NotImplementedError

    def save_checkpoint(self, epoch):
        if self.device == 0 or not dist.is_initialized():
            model_state_dict = self.model_ema.state_dict() if self.ema_decay != 0. else self.model.state_dict()
            ckpt_dict = {'epoch': epoch, 'model_state_dict': model_state_dict, 'loss': self.loss_fn,
                         'optimizer_state_dict': self.optimizer.state_dict()}
            torch.save(ckpt_dict, f'{self.ckpt_dir}/ckpt-{(epoch + 1) // self.save_freq}')

            existing_ckpts = os.listdir(self.ckpt_dir)

            if len(existing_ckpts) > self.save_ckpts:
                min_ckpt_idx = min([int(ckpt_path.split('-')[-1]) for ckpt_path in existing_ckpts])
                os.remove(f'{self.ckpt_dir}/ckpt-{min_ckpt_idx}')

    def load_checkpoint(self, map_location=None):
        if self.ckpt_num is None:
            self.ckpt_num = max([int(ckpt_path.split('-')[-1]) for ckpt_path in os.listdir(self.ckpt_dir)])

        ckpt_path = f'{self.ckpt_dir}/ckpt-{self.ckpt_num}'
        assert os.path.exists(ckpt_path), f'checkpoint path {ckpt_path} does NOT exist.'
        ckpt = torch.load(ckpt_path, map_location=map_location)

        self.model = load_parameters(self.model, ckpt)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.optimizer.param_groups[0]['capturable'] = True  # For Adam or AdamW
        self.loss_fn = ckpt['loss']
        self.epochs_trained = ckpt['epoch'] + 1

        assert self.epochs_trained == self.ckpt_num * self.save_freq, 'Incorrect trained epochs!'
        self.logger.info(f'Model restored from {ckpt_path}.\n')

    def compute_loss(self, model_out, label):
        pred, _ = model_out
        pred_loss = self.loss_fn(pred, label)
        reg_loss = self.weight_decay * sum([(para ** 2).sum() for para in self.model.parameters()])

        return pred_loss + reg_loss, torch.stack(pred_loss)

    def train_step(self, data, label):
        model_out = self.model(data, label)  # pred, pred_softmax = model_out
        loss, aux_loss = self.compute_loss(model_out, label)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        gradient_norm = tensor2numpy(self.compute_grad_norm()) if self.grad_norm else None
        self.train_metrics.update(
            loss.item(), tensor2numpy(model_out[1]), tensor2numpy(label), tensor2numpy(aux_loss), gradient_norm)

    def train_epoch(self):
        self.model.train()

        for data, label in self.train_dataloader:
            data, label = data.to(self.device), label.to(self.device)
            self.train_step(data, label)

        if self.ema_decay != 0.:
            self.update_ema_weights()

    def test_step(self, data, label):
        model_out = self.model(data, label)
        loss = self.loss_fn(model_out[0], label)

        self.test_metrics.update(loss, model_out[1], label)

    def test_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for data, label in self.test_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                self.test_step(data, label)

    def update_ema_weights(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                # p_ema.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
                p_ema.copy_(p.lerp(p_ema, self.ema_decay))

            for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
                b_ema.copy_(b)

    def compute_grad_norm(self):
        grad_norm = []

        for param in self.model.module.parameters():
            grad_norm.append(
                torch.norm(param.grad.data) if param.grad is not None else torch.tensor(0., device=self.device))

        return torch.stack(grad_norm)

    def eval_epoch(self):
        model = self.model_ema if self.ema_decay != 0. else self.model.eval()
        model = model.module if dist.is_initialized() else model
        spk_encoder = model.spk_model.spk_encoder

        if hasattr(spk_encoder.emb_layer.fc0, 'act'):
            spk_encoder = torch.nn.Sequential(spk_encoder[:-1], spk_encoder.emb_layer.fc0.linear)

        def extract_emb(eval_dataloader):
            emb = []

            with torch.no_grad():
                for data in eval_dataloader[self.device]:
                    data = data.to(self.device)

                    if 'former' in model.name:
                        offset, eval_len = 0, 16000 * 10
                        # if data.shape[1] - eval_len > 0:
                        #     offset = torch.randint(data.shape[1] - eval_len, (1,)).item()
                        data = data[:, offset: offset + eval_len]
                    emb.append(spk_encoder(data))

                if dist.is_initialized():
                    emb = dist_concat_all_gather(torch.concat(emb, dim=0))
                else:
                    emb = torch.concat(emb, dim=0)

            return emb.cpu().numpy()

        test_emb = length_norm(extract_emb(self.eval_dataloader['test']))

        if 'vox' in self.train_dataloader.dataset.source:
            test_emb = test_emb[:4874]
            enroll_emb = test_emb.copy()
        else:
            test_emb = test_emb[:17777]
            enroll_emb = length_norm(extract_emb(self.eval_dataloader['enroll']))

        # The following violates the SV evaluation protocol because test_emb.mean(0) is not available at test time
        # for debug only
        scorer = CosineScorer(
            enroll_emb - enroll_emb.mean(0), test_emb - test_emb.mean(0), self.enroll_ids, self.test_ids,
            trials_file=self.trials_file)
        scores = scorer.score()
        eer0, minDCFs0 = eval_performance(scores, self.trials_file, [0.01], c_miss=1, c_fa=1)

        scorer = CosineScorer(enroll_emb, test_emb, self.enroll_ids, self.test_ids, trials_file=self.trials_file)
        scores = scorer.score()
        eer, minDCFs = eval_performance(scores, self.trials_file, [0.01], c_miss=1, c_fa=1)

        snorm_emb = model.spk_model.spk_cls_head.weight.detach().cpu().numpy()
        snorm_emb = length_norm(snorm_emb - snorm_emb.mean(0))

        scores = scorer.score(X_enroll_in=snorm_emb, X_test_in=snorm_emb, is_snorm=True, n_top=150)
        eer_snorm, minDCFs_snorm = eval_performance(scores, self.trials_file, [0.01], c_miss=1, c_fa=1)

        return eer, minDCFs[0], eer_snorm, minDCFs_snorm[0], eer0, minDCFs0[0]

    def train(self):
        self.logger.info(f'No. of total epochs: {self.epochs}, No. of trained epochs: {self.epochs_trained}\n')

        for epoch in range(self.epochs_trained, self.epochs):
            ts = perf_counter()
            self.logger.info(f'Epoch: {epoch}/{self.epochs}')

            # Set learning rate
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr_scheduler.get_lr(epoch)

            self.logger.info(f'lr at epoch {epoch}: {self.optimizer.param_groups[0]["lr"]:.8f}')

            # Train
            self.train_dataloader.batch_sampler.set_epoch(epoch) if self.train_dataloader.batch_sampler else None
            self.train_epoch()

            train_loss, train_acc, aux_loss = self.train_metrics.result(epoch)
            aux_loss = ', '.join([f'aux_loss{i}: {aux_loss[i]:.3f}' for i in range(len(aux_loss))])
            self.logger.info(f'train_loss: {train_loss:.3f}, train_acc: {100 * train_acc:.2f}%, {aux_loss}')
            self.train_metrics.reset()

            # Test
            if self.test_dataloader is not None:
                self.test_epoch()
                test_loss, test_acc, _ = self.test_metrics.result()
                self.logger.info(f'test_loss: {test_loss:.3f}, test_acc: {100 * test_acc:.2f}%')
                self.test_metrics.reset()

            # Evaluation
            if self.eval_dataloader is not None:
                eer, min_dcf, eer_snorm, min_dcf_snorm, eer0, min_dcf0 = self.eval_epoch()
                self.logger.info(f'EER: {eer * 100:.2f}%, minDCF: {min_dcf:.3f}, '
                                 f'EER_sn: {eer_snorm * 100:.2f}%, minDCF_sn: {min_dcf_snorm:.3f}, '
                                 f'EER0: {eer0 * 100:.2f}%, minDCF0: {min_dcf0:.3f}')

            # Save checkpoints
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch)

            self.logger.info(f'Elapsed time of training epoch {epoch}: {perf_counter() - ts:.2f} s.\n')

        if dist.is_initialized():
            dist.destroy_process_group()

        self.logger.info('[*****] Training finished.\n\n\n')


class LRScheduler(object):
    def __init__(self, lr_conf='lin_1_cos_1:0.1@0,0.01@40,0.001@70'):
        mode, lr_conf = lr_conf.split(':')
        mode = mode.split('_')
        assert len(mode) % 2 == 0, 'Length of mode must be EVEN!'

        self.mode = np.concatenate([[mode[2 * i]] * int(mode[2 * i + 1]) for i in range(len(mode) // 2)])
        self.lrs = [float(lr_.split('@')[0]) for lr_ in lr_conf.split(',')]
        self.milestones = [float(lr_.split('@')[1]) for lr_ in lr_conf.split(',')]
        assert len(self.lrs) == len(self.milestones) == len(self.mode) + 1, 'Misconfig between lrs and modes!'

    @staticmethod
    def linear_lr(start_lr, end_lr, start_epoch, end_epoch, epoch):
        return start_lr + (epoch - start_epoch) * (end_lr - start_lr) / (end_epoch - start_epoch)

    @staticmethod
    def cos_lr(start_lr, end_lr, start_epoch, end_epoch, epoch):
        return start_lr + 0.5 * (end_lr - start_lr) * \
            (1 - math.cos(math.pi * (epoch - start_epoch) / (end_epoch - start_epoch)))

    @staticmethod
    def exp_lr(start_lr, end_lr, start_epoch, end_epoch, epoch):
        gama = (end_lr / start_lr) ** (1 / (end_epoch - start_epoch))

        return start_lr * gama ** (epoch - start_epoch)

    def get_lr(self, epoch):
        lr = self.lrs[-1]

        for i in range(len(self.mode)):
            if self.milestones[i] <= epoch < self.milestones[i + 1]:
                if self.mode[i] == 'cos':
                    lr = self.cos_lr(self.lrs[i], self.lrs[i + 1], self.milestones[i], self.milestones[i + 1], epoch)
                elif self.mode[i] == 'lin':
                    lr = self.linear_lr(self.lrs[i], self.lrs[i + 1], self.milestones[i], self.milestones[i + 1], epoch)
                elif self.mode[i] == 'exp':
                    lr = self.exp_lr(self.lrs[i], self.lrs[i + 1], self.milestones[i], self.milestones[i + 1], epoch)
                else:
                    lr = self.lrs[i]
                break

        return lr


class Metrics(object):
    def __init__(self, grad_norm_dir=None):
        self.grad_norm_dir = grad_norm_dir

        self.loss, self.aux_loss = 0., None
        self.n_batches, self.n_samples, self.n_correct = 0, 0, 0
        self.grad_norm = None

        if grad_norm_dir is not None:
            Path(f'{grad_norm_dir}').mkdir(parents=True, exist_ok=True)

    def reset(self):
        self.loss, self.aux_loss = 0., None
        self.n_batches, self.n_samples, self.n_correct = 0, 0, 0
        self.grad_norm = None

    def update(self, loss, preds, labels, aux_loss=None, grad_norm=None):
        assert not torch.isnan(loss), f'loss is NaN after {self.n_batches} iterations, quit training!\n\n\n'

        self.loss += loss
        self.n_correct += np.equal(preds.argmax(1), labels).sum()
        self.n_batches += 1
        self.n_samples += labels.shape[0]

        if aux_loss is not None:
            self.aux_loss = aux_loss if self.aux_loss is None else self.aux_loss + aux_loss

        if grad_norm is not None:
            self.grad_norm = grad_norm if self.grad_norm is None else self.grad_norm + grad_norm

    def result(self, epoch=None):
        # Get average (loss, acc, aux_loss) per epoch
        if self.n_batches == 0:
            raise ValueError
        else:
            ave_loss, ave_acc = self.loss / self.n_batches, self.n_correct / self.n_samples

            if self.aux_loss is None:
                ave_aux_loss = (0.,)
            else:
                ave_aux_loss = tuple([loss_ / self.n_batches for loss_ in self.aux_loss])

            if self.grad_norm is not None:
                np.save(f'{self.grad_norm_dir}/{epoch}.npy', self.grad_norm / self.n_batches)

        return ave_loss, ave_acc, ave_aux_loss


def tensor2numpy(tensor):
    if tensor.dim() > 0:
        return tensor.detach().cpu().numpy()
    return tensor.item()


def load_parameters(model, ckpt):
    ckpt_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items() if 'module.' in k}
    model_state_dict = model.state_dict()
    ckpt_state_dict_new = {}

    # make the ckpt_state_dict keys the same as the model keys in case that the model keys may be changed
    for ckpt_key, model_key in zip(ckpt_state_dict, model_state_dict):
        ckpt_state_dict_new[model_key] = ckpt_state_dict[ckpt_key]

    model.load_state_dict(ckpt_state_dict_new)

    return model
