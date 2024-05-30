import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')

class NewIRNModel(BaseModel):
    def __init__(self, opt):
        super(NewIRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG_down = networks.define_G(opt).to(self.device)
        self.netG_up = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG_down = DistributedDataParallel(self.netG_down, device_ids=[torch.cuda.current_device()])
            self.netG_up = DistributedDataParallel(self.netG_up, device_ids=[torch.cuda.current_device()])
        else:
            self.netG_down = DataParallel(self.netG_down)
            self.netG_up = DataParallel(self.netG_up)
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG_down.train()
            self.netG_up.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd_G = float(train_opt['weight_decay_G']) if train_opt['weight_decay_G'] else 0
            optim_params_down = [v for k, v in self.netG_down.named_parameters() if v.requires_grad]
            optim_params_up = [v for k, v in self.netG_up.named_parameters() if v.requires_grad]

            self.optimizer_G_down = torch.optim.Adam(optim_params_down, lr=train_opt['lr_G'],
                                                     weight_decay=wd_G,
                                                     betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_G_up = torch.optim.Adam(optim_params_up, lr=train_opt['lr_G'],
                                                   weight_decay=wd_G,
                                                   betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G_down)
            self.optimizers.append(self.optimizer_G_up)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_H = data['GT'].to(self.device)  # High-Quality (HQ) images
        if data.get('LQ') is not None:
            self.ref_L = data['LQ'].to(self.device)  # Low-Quality (LQ) images
        else:
            self.ref_L = self.downscale(self.real_H)  # Generate LQ images if not provided

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]
        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG_up(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)
        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer_G_down.zero_grad()
        self.optimizer_G_up.zero_grad()

        # forward downscaling
        self.input = self.real_H
        self.output = self.netG_down(x=self.input)

        zshape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L.detach()

        l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

        # backward upscaling
        LR = self.Quantization(self.output[:, :3, :, :])

        if self.train_opt['add_noise_on_y']:
            probability = self.train_opt['y_noise_prob']
            noise_scale = self.train_opt['y_noise_scale']
            prob = np.random.rand()
            if prob < probability:
                LR = LR + noise_scale * self.gaussian_batch(LR.shape)

        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] is not None else 1
        y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        l_back_rec = self.loss_backward(self.real_H, y_)

        # total loss
        loss = l_forw_fit + l_back_rec + l_forw_ce
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG_down.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.netG_up.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G_down.step()
        self.optimizer_G_up.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_ce'] = l_forw_ce.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def test(self):
        Lshape = self.ref_L.shape
        input_dim = Lshape[1]
        self.input = self.real_H
        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]
        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] is not None:
            gaussian_scale = self.test_opt['gaussian_scale']
        self.netG_down.eval()
        self.netG_up.eval()
        with torch.no_grad():
            self.forw_L = self.netG_down(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG_up(x=y_forw, rev=True)[:, :3, :, :]
        self.netG_down.train()
        self.netG_up.train()

    def downscale(self, HR_img):
        self.netG_down.eval()
        with torch.no_grad():
            LR_img = self.netG_down(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(LR_img)
        self.netG_down.train()
        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
        self.netG_up.eval()
        with torch.no_grad():
            HR_img = self.netG_up(x=y_, rev=True)[:, :3, :, :]
        self.netG_up.train()
        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG_down)
        if isinstance(self.netG_down, nn.DataParallel) or isinstance(self.netG_down, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG_down.__class__.__name__, self.netG_down.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG_down.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G (downscaling) structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        s, n = self.get_network_description(self.netG_up)
        if isinstance(self.netG_up, nn.DataParallel) or isinstance(self.netG_up, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG_up.__class__.__name__, self.netG_up.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG_up.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G (upscaling) structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G_down = self.opt['path']['pretrain_model_G_down']
        load_path_G_up = self.opt['path']['pretrain_model_G_up']
        if load_path_G_down is not None:
            logger.info('Loading model for G_down [{:s}] ...'.format(load_path_G_down))
            self.load_network(load_path_G_down, self.netG_down, self.opt['path']['strict_load'])
        if load_path_G_up is not None:
            logger.info('Loading model for G_up [{:s}] ...'.format(load_path_G_up))
            self.load_network(load_path_G_up, self.netG_up, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG_down, 'G_down', iter_label)
        self.save_network(self.netG_up, 'G_up', iter_label)
