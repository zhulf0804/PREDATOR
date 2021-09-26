import numpy as np
import os
import shutil
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import ThreeDMatch, get_dataloader
from models import architectures, PREDATOR
from losses import PREDATORLoss
from utils import decode_config, setup_seed

CUR = os.path.dirname(os.path.abspath(__file__))


def save_summary(writer, loss_dict, global_step, tag, lr=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)


def main():
    setup_seed(1234)
    config = decode_config(os.path.join(CUR, 'configs', 'threedmatch.yaml'))
    config = edict(config)
    config.architecture = architectures[config.dataset]

    saved_path = config.exp_dir
    saved_ckpt_path = os.path.join(saved_path, 'checkpoints')
    saved_logs_path = os.path.join(saved_path, 'summary')
    os.makedirs(saved_path, exist_ok=True)
    os.makedirs(saved_ckpt_path, exist_ok=True)
    os.makedirs(saved_logs_path, exist_ok=True)
    shutil.copyfile(os.path.join(CUR, 'configs', 'threedmatch.yaml'), 
                    os.path.join(saved_path, 'threedmatch.yaml'))

    train_dataset = ThreeDMatch(root=config.root,
                                split='train',
                                aug=True,
                                overlap_radius=config.overlap_radius)
    val_dataset = ThreeDMatch(root=config.root,
                                split='val',
                                aug=False,
                                overlap_radius=config.overlap_radius)
    train_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                           dataset=train_dataset,
                                                           batch_size=config.batch_size,
                                                           num_workers=config.num_workers,
                                                           shuffle=True,
                                                           neighborhood_limits=None)
    val_dataloader, _ = get_dataloader(config=config,
                                       dataset=val_dataset,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       neighborhood_limits=neighborhood_limits)

    print(neighborhood_limits)
    model = PREDATOR(config).cuda()
    predator_loss = PREDATORLoss(config)

    if config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )

    # create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.scheduler_gamma,
    )

    writer = SummaryWriter(saved_logs_path)

    w_saliency = config.w_saliency_loss
    w_saliency_update = False
    best_recall, best_circle_loss = 0, 1e8

    for epoch in range(config.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for inputs in tqdm(train_dataloader):
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda()
                else:
                    inputs[k] = inputs[k].cuda()

            optimizer.zero_grad()

            batched_feats = model(inputs)
            stack_points = inputs['points']
            stack_lengths = inputs['stacked_lengths']
            coords_src = stack_points[0][:stack_lengths[0][0]]
            coords_tgt = stack_points[0][stack_lengths[0][0]:]
            feats_src = batched_feats[:stack_lengths[0][0]]
            feats_tgt = batched_feats[stack_lengths[0][0]:]
            coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
            transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1

            loss_dict = predator_loss(coords_src=coords_src,
                                      coords_tgt=coords_tgt,
                                      feats_src=feats_src,
                                      feats_tgt=feats_tgt,
                                      coors=coors,
                                      transf=transf,
                                      w_saliency=w_saliency)

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % config.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'])
            train_step += 1

        torch.cuda.empty_cache()
        scheduler.step()

        total_circle_loss, total_recall = [], []
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(val_dataloader):
                for k, v in inputs.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            inputs[k][i] = inputs[k][i].cuda()
                    else:
                        inputs[k] = inputs[k].cuda()
                batched_feats = model(inputs)
                stack_points = inputs['points']
                stack_lengths = inputs['stacked_lengths']
                coords_src = stack_points[0][:stack_lengths[0][0]]
                coords_tgt = stack_points[0][stack_lengths[0][0]:]
                feats_src = batched_feats[:stack_lengths[0][0]]
                feats_tgt = batched_feats[stack_lengths[0][0]:]
                coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
                transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1

                loss_dict = predator_loss(coords_src=coords_src,
                                          coords_tgt=coords_tgt,
                                          feats_src=feats_src,
                                          feats_tgt=feats_tgt,
                                          coors=coors,
                                          transf=transf,
                                          w_saliency=w_saliency)

                circle_loss = loss_dict['circle_loss']
                total_circle_loss.append(circle_loss.detach().cpu().numpy())
                recall = loss_dict['recall']
                total_recall.append(recall.detach().cpu().numpy())

                global_step = epoch * len(val_dataloader) + val_step + 1

                if global_step % config.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1

        # print('Epoch: ', epoch,  loss_dict)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, f'{epoch+1}.pth'))
        if np.mean(total_circle_loss) < best_circle_loss:
            best_circle_loss = np.mean(total_circle_loss)
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, 'best_loss.pth'))
        if np.mean(total_recall) > best_recall:
            best_recall = np.mean(total_recall)
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, 'best_recall.pth'))
        if not w_saliency_update and np.mean(total_recall) > 0.3:
            w_saliency_update = True
            w_saliency = 1

        torch.cuda.empty_cache()
        model.train()


if __name__ == '__main__':
    main()
