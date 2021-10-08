# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 12345))
# ptvsd.wait_for_attach()
import os
import copy
import argparse
import datetime
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms, utils

from networks import UNet, GammaNet, eta_to_gamma
from diffusion import GaussianDiffusion
from utils.distributed import get_rank, synchronize, get_world_size, data_sampler, dist_collect
from utils.misc import calc_grad_norm, EMA, cycle, unwrap_module
try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def is_tensorboard_available():
    return _has_tensorboard

class Dataset(data.Dataset):
    EXTS = ['jpg', 'jpeg', 'png']
    def __init__(self, folder_path, transform):
        super().__init__()
        self.folder_path = folder_path
        self.paths = [p for ext in self.EXTS for p in folder_path.glob(f'**/*.{ext}')]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



# trainer class
class Trainer(object):
    def __init__(self, args, dataset, diffusion, model, gammanet):
        self.args = args
        self.device = args.device
        self.num_timesteps = diffusion.num_timesteps

        self.diffusion, self.model, self.gammanet = diffusion, model, gammanet
        self.model.to(self.device)
        self.gammanet.to(self.device)
        self.ema = EMA(args.ema_decay)
        self.ema_model, self.ema_gammanet = copy.deepcopy(self.model), copy.deepcopy(self.gammanet)

        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) +
                                      list(self.gammanet.parameters()),
                                      betas=(0.9, 0.99),
                                      lr=args.lr, weight_decay=0.01)

        self.step = 0
        if args.resume:
            self.load(file_path=args.resume)

        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            self.gammanet = torch.nn.parallel.DistributedDataParallel(
                self.gammanet,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        self.master_proc = False
        if get_rank() == 0:
            self.master_proc = True

        self.tb_writer = None
        if self.master_proc:
            trainid = str(datetime.datetime.now())
            trainid = trainid[:trainid.rfind(":")].replace("-", "").replace(":", "").replace(" ", "_")
            self.output_dir = Path(args.output_dir) / trainid
            (self.output_dir / 'sample').mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'checkpoint').mkdir(parents=True, exist_ok=True)

            if args.tb_writer and is_tensorboard_available():
                self.tb_writer = SummaryWriter(self.output_dir / 'tb_log')

        self.dataset = dataset
        self.dataloader = cycle(
            data.DataLoader(self.dataset,
                            batch_size=args.batch_size,
                            sampler=data_sampler(self.dataset,
                                                 shuffle=True,
                                                 distributed=args.distributed),
                            pin_memory=True,
                            num_workers=args.num_workers))


    def reset_ema(self):
        self.ema_model.load_state_dict(self.model_core.state_dict())
        if self.with_ema_eta:
            self.ema_eta.load_state_dict(self.eta_core.state_dict())

    def update_ema(self, step):
        step_start_ema = self.args.step_start_ema
        if step < step_start_ema:
            self.ema.reset_model_lst((self.ema_model, self.ema_gammanet), (self.model, self.gammanet))
            return
        self.ema.update_model_lst((self.ema_model, self.ema_gammanet), (self.model, self.gammanet))


    def save(self, step):
        data = {
            'step': step,
            'model': unwrap_module(self.model).state_dict(),
            'gammanet': unwrap_module(self.gammanet).state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'ema_gammanet': self.ema_gammanet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(data, self.output_dir / 'checkpoint' / f'model-{step:08d}.pt')

    def load(self, file_path):
        data = torch.load(file_path, map_location='cpu')

        self.step = data['step']
        self.gammanet.load_state_dict(data['gammanet'])
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema_model'])
        self.ema_gammanet.load_state_dict(data['ema_gammanet'])
        self.optimizer.load_state_dict(data['optimizer'])

    @torch.no_grad()
    def visualize_eta(self, step):
        assert self.master_proc
        self.ema_gammanet.eval()

        ti = torch.linspace(0, 1, self.num_timesteps+1, device=self.device).view(-1, 1)
        eta_t, eta_be, gamma_be = self.ema_gammanet(ti)
        gamma_t, _ = eta_to_gamma(eta_t, eta_be, gamma_be)
        ti = ti.view(-1).cpu().numpy()
        gamma_t = gamma_t.view(-1).cpu().numpy()
        plt.figure()
        plt.plot(ti, -gamma_t)  # logsnr_t = - gamma_t
        # plt.plot(ti, np.exp(-gamma_t))
        plt.savefig(self.output_dir / 'sample' / f'eta-{step:08d}.png')
        plt.close()

    @torch.no_grad()
    def visualize(self, step, image_shape, num_vis=64, clip_value=1):

        self.ema.broadcast(self.ema_model, self.ema_gammanet)
        self.ema_model.eval()
        self.ema_gammanet.eval()

        num_vis_per_gpu = num_vis // get_world_size()
        x = self.diffusion.p_sample_loop(
            self.ema_model,
            self.ema_gammanet,
            shape=(num_vis_per_gpu, *image_shape),
            clip_value=clip_value,
            device=self.device,
            progress=(True if self.master_proc else False),
        )
        x = dist_collect(x)

        if self.master_proc:
            utils.save_image(x,
                             fp=(self.output_dir / 'sample' / f'sample-{step:08d}.png'),
                             normalize=True,
                             value_range=(-1, 1))

    def train(self):
        args = self.args

        pbar = range(args.train_num_steps)
        if self.master_proc:
            pbar = tqdm(pbar, initial=self.step, dynamic_ncols=True, smoothing=0.01)

        for step in pbar:

            loss_dict = {}
            for i in range(args.gradient_accumulate_every):
                x = next(self.dataloader)
                x = x.to(self.device)

                if args.random_t_vdm:
                    t = torch.randint(0, self.diffusion.num_timesteps, (1, )) + torch.arange(x.size(0))
                    t = (t % self.diffusion.num_timesteps + 1).to(self.device)
                else:
                    t = torch.randint(1, self.diffusion.num_timesteps + 1, (x.size(0),), device=self.device).long()

                terms = self.diffusion.training_losses(self.model, self.gammanet, x, t)
                loss = (terms['diff'] + terms['prior'] + terms['rec']) / args.gradient_accumulate_every
                loss.backward()

                # for logging
                for k, v in terms.items():
                    v = v.detach().mean()
                    try:
                        loss_dict[k].append(v)
                    except KeyError:
                        loss_dict[k] = [v]

            if args.gnorm_clip_model > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.gammanet.parameters()), args.gnorm_clip_model)
            else:
                gnorm = calc_grad_norm(list(self.model.parameters()) + list(self.gammanet.parameters()))

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.master_proc:
                self.update_ema(step)

            if step % args.vis_every == 0 and step > 0:
                self.visualize(step, image_shape=x.shape[1:], num_vis=args.vis_num, )

            if self.master_proc:
                loss_step = {k: sum(v).item() for k, v in loss_dict.items()}
                msg = [
                    *[f"{k}: {v:7.4g}" for k, v in loss_step.items()],
                    f"gnorm: {gnorm:.3e}",
                ]
                pbar.set_description(", ".join(msg))

                if self.tb_writer:
                    self.tb_writer.add_scalars('loss', loss_step, step)
                    self.tb_writer.add_scalar('gnorm', gnorm, step)

                if step % args.vis_eta_every == 0 and step > 0:
                    self.visualize_eta(step)

                if step % args.save_every == 0 and step > 0:
                    self.save(step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--local_rank', type=int, default=0)
    # dataset
    parser.add_argument('--data_dir', help='path of image data.')
    parser.add_argument('--image_size', default=128, type=int)
    # training
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--step_start_ema', default=2000, type=int, help='from when ema accumulation starts.')
    parser.add_argument('--gradient_accumulate_every', default=1, type=int, help='gradient accumulation steps.')
    parser.add_argument('--ema_decay', default=0.999, type=float, help='exponential moving average decay.')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--train_num_steps', default=1_000_000, type=int, help='total training steps.')
    parser.add_argument('--random_t_vdm', default=False, type=bool)
    parser.add_argument('--gnorm_clip_model', default=1, type=float)
    # log ckpt and visualize
    parser.add_argument('--output_dir', default='experiment', type=str, help='path of output root.')
    parser.add_argument('--tb_writer', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--vis_eta_every', default=1000, type=int)
    parser.add_argument('--vis_every', default=5000, type=int)
    parser.add_argument('--save_every', default=10000, type=int)
    parser.add_argument('--vis_num', default=64, type=int)
    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    unet_setting = {
        'in_channel': 3,
        'channel': 128,
        'channel_multiplier': [1, 1, 2, 2, 4, 4],
        'n_res_blocks': 2,
        'attn_strides': [16],
        'attn_heads': 1,
        'use_affine_time': False,
        'dropout': 0.0,
        'fold': 1,
        'cond_src': 'eta_norm',
        'max_freq': 10,
    }
    gammanet_setting = {
        'd_hidden': 1024,
        'gamma_0': -10.0,
        'gamma_1': 10.0,
    }
    diffusion_setting = {
        'num_timesteps': 1000,
        'match_obj': 'eps',  # eps, x_start
    }

    model = UNet(**unet_setting)
    gammanet = GammaNet(**gammanet_setting)
    diffusion = GaussianDiffusion(**diffusion_setting)

    dataset = Dataset(Path(args.data_dir).expanduser(),
                      transform=transforms.Compose([
                          transforms.Resize(args.image_size),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(args.image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5),
                                               inplace=True),
                      ]))

    trainer = Trainer(args, dataset, diffusion, model, gammanet)
    trainer.train()
