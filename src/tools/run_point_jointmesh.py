
# Modified from METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------
"""
Training and evaluation codes for
3D hand pose estimation from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch

from torchvision.utils import make_grid
import numpy as np
import cv2

from src.modeling.model.modeling_AHRNRT import FastMETRO_Hand_Network as AHR_Network
from src.modeling._mano import MANO, Mesh

import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader
from src.utils.logger import setup_logger
from src.utils.comm import is_main_process, get_rank, get_world_size
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.geometric_layers import orthographic_projection
from src.tools.eval import main as ping
from src.tools.loss1 import calc_losses
from src.tools.run_hand_gai import visualize_mesh1
from src.utils.renderer_opendr import OpenDR_Renderer, visualize_reconstruction_opendr, \
    visualize_reconstruction_multi_view_opendr

try:
    from src.utils.renderer_pyrender import PyRender_Renderer, visualize_reconstruction_pyrender, \
        visualize_reconstruction_multi_view_pyrender
except:
    print("Failed to import renderer_pyrender. Please see docs/Installation.md")

from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for _ in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir



def run_train(args, train_dataloader,  _AHRNET, mano_model, mesh_sampler, renderer):
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    iteration = args.resume_epoch * iters_per_epoch

    #
    train_writer = SummaryWriter(os.path.join('./log_output', 'train'))

    #

    _AHRNET_model_without_ddp = _AHRNET
    if args.distributed:
        _AHRNET_model = torch.nn.parallel.DistributedDataParallel(
            _AHRNET, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        _AHRNET_model_without_ddp = _AHRNET_model_without_ddp.module
        if is_main_process():
            logger.info(
                ' '.join(
                    ['Local-Rank: {o}', 'Max-Iteration: {a}', 'Iterations-per-Epoch: {b}',
                     'Number-of-Training-Epochs: {c}', ]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    param_dicts = [
        {"params": [p for p in  _AHRNET_model_without_ddp.parameters() if p.requires_grad]}
    ]

    # optimizer & learning rate scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    criterion_heatmap = torch.nn.MSELoss().cuda(args.device)

    MAP = torch.eye(112 * 112)

    start_training_time = time.time()
    end = time.time()

    _AHRNET_model.train()

    # add time
    batch_time = AverageMeter()
    data_time = AverageMeter()

    log_losses = AverageMeter()
    log_loss_3d_joints = AverageMeter()
    log_loss_3d_vertices = AverageMeter()
    log_loss_2d_joints = AverageMeter()
    #htloss
    log_meshhtloss = AverageMeter()
    log_jothtloss = AverageMeter()


    for _, (img_keys, images, annotations) in enumerate(train_dataloader):
        _AHRNET.train()
        iteration = iteration + 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)

        images = images.cuda(args.device)  # batch_size X 3 X 224 X 224

        # gt 2d joints
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_mesh = annotations['has_smpl'].cuda(args.device)
        has_3d_joints = has_mesh.clone()
        has_2d_joints = has_mesh.clone()


        # generate mesh
        gt_3d_vertices_fine, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        gt_3d_vertices_fine = gt_3d_vertices_fine / 1000.0
        gt_3d_joints = gt_3d_joints / 1000.0
        gt_3d_vertices_coarse = mesh_sampler.downsample(gt_3d_vertices_fine)
        gt_3d_vertices_coarse1 = mesh_sampler.downsample(gt_3d_vertices_fine, n1=0, n2=2)


        # normalize gt based on hand's wrist
        gt_3d_root = gt_3d_joints[:, cfg.J_NAME.index('Wrist'), :]
        gt_3d_vertices_fine = gt_3d_vertices_fine - gt_3d_root[:, None, :]
        gt_3d_vertices_coarse = gt_3d_vertices_coarse - gt_3d_root[:, None, :]
        gt_3d_vertices_coarse1 = gt_3d_vertices_coarse1 - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1], 4)).cuda(args.device)
        gt_3d_joints_with_tag[:, :, :3] = gt_3d_joints

        # forward-pass
        out =  _AHRNET(images)
        pred_cam, pred_3d_joints_from_token = out['pred_cam'], out['pred_3d_joints']
        pred_3d_vertices_coarse, pred_3d_vertices_fine = out['pred_3d_vertices_coarse'], out['pred_3d_vertices_fine']

        heatmap_ = out['heatmap_vert']
        heatmap1_ = out['heatmap_joint']
        need_hloss = True
        pred_2d_joints_from_mano, pred_3d_vertices_fine, loss_2d_joints, loss_3d_joints, loss_vertices, loss, mesh_htloss, jot_htloss = calc_losses(args, pred_cam,
                                                                                                    pred_3d_joints_from_token,
                                                                                                    pred_3d_vertices_coarse,
                                                                                                    pred_3d_vertices_fine,
                                                                                                    gt_3d_vertices_coarse,
                                                                                                    gt_3d_vertices_fine,
                                                                                                    gt_3d_joints_with_tag,
                                                                                                    gt_2d_joints,
                                                                                                    has_3d_joints,
                                                                                                    has_2d_joints,
                                                                                                    has_mesh,
                                                                                                    criterion_keypoints,
                                                                                                    criterion_2d_keypoints,
                                                                                                    criterion_vertices,
                                                                                                    mano_model,
                                                                                                    heatmap_,
                                                                                                    heatmap1_,
                                                                                                    criterion_heatmap,
                                                                                                    MAP, need_hloss)


        # update logs
        log_loss_3d_joints.update(loss_3d_joints.item(), batch_size)
        log_loss_3d_vertices.update(loss_vertices.item(), batch_size)
        log_loss_2d_joints.update(loss_2d_joints.item(), batch_size)
        #htloss
        log_meshhtloss.update(mesh_htloss.item(), batch_size)
        log_jothtloss.update(jot_htloss.item(), batch_size)

        log_losses.update(loss.item(), batch_size)

        # back-propagation
        optimizer.zero_grad()
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(_AHRNET.parameters(), args.clip_max_norm)
        optimizer.step()


        # add time
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            logger.info(
                ' '.join(
                    ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                ).format(eta=eta_string, ep=epoch, iter=iteration,
                         memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                + 'loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f},meshht loss: {:.4f},jotht loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2d_joints.avg, log_loss_3d_joints.avg, log_loss_3d_vertices.avg,log_meshhtloss.avg,log_jothtloss.avg,
                    batch_time.avg,
                    data_time.avg,
                    optimizer.param_groups[0]['lr'])
            )

            train_writer.add_scalar('Loss/loss', log_losses.avg, iteration)
            train_writer.add_scalar('Loss/loss_3d_joints', log_loss_3d_joints.avg, iteration)
            train_writer.add_scalar('Loss/loss_3d_vertices', log_loss_3d_vertices.avg, iteration)
            train_writer.add_scalar('Loss/loss_2d_joints', log_loss_2d_joints.avg, iteration)
            train_writer.add_scalar('Loss/mloss', log_losses.avg, epoch)
            train_writer.add_scalar('Loss/meshht', log_meshhtloss.avg, iteration)
            train_writer.add_scalar('Loss/jotht', log_jothtloss.avg, iteration)

            # end################################################################
            # visualize estimation results during training
            if args.visualize_training and (iteration >= args.logging_steps):
                visual_imgs = visualize_mesh1(renderer,
                                              annotations['ori_img'].detach(),
                                              annotations['joints_2d'].detach(),
                                              pred_3d_vertices_fine.detach(),
                                              pred_cam.detach(),
                                              pred_2d_joints_from_mano.detach())

                visual_imgs = visual_imgs.transpose(0, 1)
                visual_imgs = visual_imgs.transpose(1, 2)
                visual_imgs = np.asarray(visual_imgs)
                if is_main_process():
                    stamp = str(epoch) + '_' + str(iteration)
                    temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                    if args.use_opendr_renderer:
                        visual_imgs[:, :, ::-1] = visual_imgs[:, :, ::-1] * 255
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1]))

        # save checkpoint
        if (iteration % iters_per_epoch) == 0:
            lr_scheduler.step()
            if (epoch != 0) and ((epoch % args.saving_epochs) == 0):
                checkpoint_dir = save_checkpoint(_AHRNET, args, epoch, iteration)
                #

                args.resume_checkpoint = checkpoint_dir
                #
                val_dataloader = make_hand_data_loader(args, args.val_yaml, args.distributed, is_train=False,
                                                           scale_factor=args.img_scale_factor)
                xyz_auc, xyz_o, mesh_auc, mesh_o = run_eval_and_save(args, 'freihand', val_dataloader, _AHRNET, mano_model, renderer,
                                      mesh_sampler)

                train_writer.add_scalar('Loss/xyz_auc', xyz_auc, epoch)
                train_writer.add_scalar('Loss/xyz_o', xyz_o, epoch)
                train_writer.add_scalar('Loss/mesh_auc', mesh_auc, epoch)
                train_writer.add_scalar('Loss/mesh_o', mesh_o, epoch)


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(total_time_str, total_training_time / max_iter))
    checkpoint_dir = save_checkpoint(_AHRNET, args, epoch, iteration)


def run_eval_and_save(args, split, val_dataloader, _AHRNET, mano_model, renderer, mesh_sampler):
    if args.distributed:
        _AHRNET_model = torch.nn.parallel.DistributedDataParallel(
            _AHRNET, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    _AHRNET_model.eval()

    a = run_inference_hand_mesh(args, val_dataloader,
                            _AHRNET_model,
                            mano_model, renderer, split)
    checkpoint_dir = save_checkpoint(_AHRNET_model, args, 0, 0)

    xyz_auc, xyz_o, mesh_auc, mesh_o = ping("./freihand-master",
         a,
         "./save",
         set_name='evaluation')

    print(xyz_auc, xyz_o, mesh_auc, mesh_o)

    logger.info("The experiment completed successfully. Finalizing run...")

    return xyz_auc, xyz_o, mesh_auc, mesh_o


def run_inference_hand_mesh(args, val_loader, _AHRNET_model, mano_model, renderer, split):
    # switch to evaluate mode
    _AHRNET_model.eval()
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []

    with torch.no_grad():
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)  # batch_size X 3 X 224 X 224

            # forward-pass
            out = _AHRNET_model(images)
            pred_cam, pred_3d_vertices_fine = out['pred_cam'], out['pred_3d_vertices_fine']

            # obtain 3d joints from full mesh
            pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
            pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, cfg.J_NAME.index('Wrist'), :]

            # normalize predicted vertices
            pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
            # normalize predicted joints
            pred_3d_joints_from_mano = pred_3d_joints_from_mano - pred_3d_joints_from_mano_wrist[:, None, :]

            for j in range(batch_size):
                fname_output_save.append(img_keys[j])
                pred_3d_vertices_list = pred_3d_vertices_fine[j].tolist()
                mesh_output_save.append(pred_3d_vertices_list)
                pred_3d_joints_from_mano_list = pred_3d_joints_from_mano[j].tolist()
                joint_output_save.append(pred_3d_joints_from_mano_list)

            if args.run_eval_and_visualize:
                if (i % 20) == 0:
                    # obtain 3d joints, which are regressed from the full mesh
                    pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
                    # obtain 2d joints, which are projected from 3d joints of mesh
                    pred_2d_joints_from_mano = orthographic_projection(pred_3d_joints_from_mano.contiguous(),
                                                                       pred_cam.contiguous())

                    visual_imgs = visualize_mesh1(renderer,
                                                  annotations['ori_img'].detach(),
                                                  annotations['joints_2d'].detach(),
                                                  pred_3d_vertices_fine.detach(),
                                                  pred_cam.detach(),
                                                  pred_2d_joints_from_mano.detach())

                    visual_imgs = visual_imgs.transpose(0, 1)
                    visual_imgs = visual_imgs.transpose(1, 2)
                    visual_imgs = np.asarray(visual_imgs)

                    inference_setting = 'sc%02d_rot%s' % (int(args.sc * 10), str(int(args.rot)))
                    temp_fname = args.output_dir + 'visual_' + inference_setting + '_batch' + str(i) + '.jpg'
                    if args.use_opendr_renderer:
                        visual_imgs[:, :, ::-1] = visual_imgs[:, :, ::-1] * 255
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1]))

    print('save results to pred.json')
    with open('pred.json', 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)


    predlist=[]
    predlist.append(joint_output_save)
    predlist.append(mesh_output_save)
    return predlist


def visualize_mesh(renderer, images, pred_vertices, pred_cam):
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get predicted vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_cam[i].cpu().numpy()
        # Visualize reconstruction
        if args.use_opendr_renderer:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_opendr(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_opendr(img, vertices, cam, renderer)
        else:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_pyrender(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
        rend_img = rend_img.transpose(2, 0, 1)
        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml",
                        default='./datasets/freihand/train.yaml',
                        type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml",
                        default='./datasets/freihand/test.yaml',
                        type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='./out_save_epoch_train/', type=str,
                        required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--saving_epochs", default=5, type=int)
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_epoch", default=0, type=int)
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.3, type=float,
                        help='gradient clipping maximal norm')
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--vloss_w_full", default=0.5, type=float)
    parser.add_argument("--vloss_w_sub2", default=0.5, type=float)
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--heatmap_loss_weight", default=5, type=float)
    parser.add_argument("--heatmap1_loss_weight", default=5, type=float)

    parser.add_argument("--vertices_fine_loss_weight", default=0.50, type=float)
    parser.add_argument("--vertices_coarse_loss_weight", default=0.50, type=float)
    parser.add_argument("--edge_gt_loss_weight", default=1.0, type=float)
    parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    # Model parameters
    parser.add_argument("--model_name", default='FastMETRO-S', type=str,
                        help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)
    # Others
    #########################################################
    parser.add_argument("--run_evaluation", default=True, action='store_true', )
    parser.add_argument("--run_eval_and_visualize", default=True, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--model_save', default=False, action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--exp", default='FastMETRO', type=str, required=False)
    parser.add_argument("--visualize_training", default=True, action='store_true', )
    parser.add_argument("--visualize_multi_view", default=False, action='store_true', )
    parser.add_argument("--use_opendr_renderer", default=True, action='store_true', )
    parser.add_argument("--multiscale_inference", default=False, action='store_true', )
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument("--aml_eval", default=False, action='store_true', )

    args = parser.parse_args()
    return args


def main(args):
    print("AHRNET for 3D Hand Joint estimation!")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print("Init distributed training on local rank {} ({}), world size {}".format(args.local_rank,
                                                                                      int(os.environ["LOCAL_RANK"]),
                                                                                      args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        torch.distributed.barrier()

    mkdir(args.output_dir)
    logger = setup_logger("AHRNET", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    if args.use_opendr_renderer:
        renderer = OpenDR_Renderer(faces=mano_model.face)
    else:
        renderer = PyRender_Renderer(faces=mano_model.face)

    logger.info("Training Arguments %s", args)

    #Load model
    if args.run_evaluation and (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and (
            'state_dict' not in args.resume_checkpoint):
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _AHRNET = torch.load(args.resume_checkpoint)

    _AHRNET = AHR_Network(args, mesh_sampler)
    overall_params = sum(p.numel() for p in  _AHRNET.parameters() if p.requires_grad)
    logger.info('Number of Overall learnable parameters: {}'.format(overall_params))

    if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _AHRNET.load_state_dict(state_dict, strict=False)
        del state_dict

    _AHRNET.to(args.device)

    if args.run_evaluation:
        val_dataloader = make_hand_data_loader(args, args.val_yaml, args.distributed, is_train=False,
                                               scale_factor=args.img_scale_factor)
        run_eval_and_save(args, 'freihand', val_dataloader,  _AHRNET, mano_model, renderer, mesh_sampler)
    else:
        train_dataloader = make_hand_data_loader(args, args.train_yaml, args.distributed, is_train=True,
                                                 scale_factor=args.img_scale_factor)
        run_train(args, train_dataloader,  _AHRNET, mano_model, mesh_sampler, renderer)

        #


if __name__ == "__main__":
    args = parse_args()
    main(args)