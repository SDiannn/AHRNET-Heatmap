"""
----------------------------------------------------------------------------------------------
Modified from PointHMR
Modified from MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

import torch
import src.modeling.data.config as cfg
from src.utils.geometric_layers import orthographic_projection
from torch.nn import functional as F

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def calc_triangle_area(face):
    a,b,c = face[:,0,:], face[:,1,:], face[:,2,:]
    # print(a,b,c)
    ab, bc, ca = torch.sqrt(((a-b)**2).sum(-1)), torch.sqrt(((b-c)**2).sum(-1)), torch.sqrt(((c-a)**2).sum(-1))

    return ab+bc+ca

def surface_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):

    pred_surface_with_shape = pred_vertices[has_smpl == 1][:,smpl_face].reshape(-1,3,3)
    gt_surface_with_shape = gt_vertices[has_smpl == 1][:,smpl_face].reshape(-1,3,3)

    pred_area = calc_triangle_area(pred_surface_with_shape)
    gt_area = calc_triangle_area(gt_surface_with_shape)

    # isnan = torch.isnan(pred_area) == False
    if torch.isnan(pred_area).sum():
        print(pred_vertices)
        raise Exception("NaN")

    if len(gt_surface_with_shape) > 0:
        loss = criterion_vertices(gt_area, pred_area)
        if torch.isnan(loss).sum():
            raise Exception("loss")
        return loss
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)
def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def make_gt(verts_camed, has_smpl, MAP, img_size=112):
    verts_camed = ((verts_camed[has_smpl==1] + 1) * 0.5) * img_size
    x = verts_camed[:, :,0].long()
    y = verts_camed[:, :,1].long()

    indx = img_size*y + x
    flag1 = indx<img_size*img_size
    flag2 = -1 < indx
    flag = flag2*flag1

    GT = MAP[indx[flag]].reshape(-1,1,img_size,img_size).to(verts_camed.device)
    # GT = conv_gauss(GT, device=verts_camed.device)
    #
    # GT[GT==0] = -0.1

    return GT, flag

def conv_gauss(img,device):
    k = torch.Tensor([[ 1.25, 2.5, 1.25]])
    kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1).to(device)
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel)

def calc_heatmap_loss(heatmap,gt,has_smpl,flag, criterion_heatmap):
    pred = heatmap[has_smpl==1][flag]
    # mask = gt.float().sum(dim=2).sum(dim=1).gt(0).float()
    # loss = (((pred - gt) ** 2).mean(dim=2).mean(dim=1) * mask).sum() / mask.sum()
    # return criterion_heatmap(pred,gt)
    # print(pred.shape)
    # print(gt.shape)
    return dice_loss(pred.unsqueeze(1),gt.flatten(2))

    # return focal_loss(pred,gt)
    # return focal_loss(pred, gt)


def dice_loss(pred, target, smooth=1e-5):
    # pred = torch.sigmoid(pred)
    # binary cross entropy loss
    bce = F.binary_cross_entropy(pred, target, reduction='mean')*1e3

    # dice coefficient
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred).sum(dim=(1, 2)) + (target).sum(dim=(1, 2))
    dice = 2.0 * (intersection + smooth) / (union + 2 * smooth)

    # dice loss
    dice_loss = 1.0 - dice
    # total loss
    loss = dice_loss.mean() + bce

    return loss

def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = torch.zeros(gt.size(0)).to(pred.device)

    # log(0) lead to nan loss, collipsed
    pred_log = pred.clone()
    pred_log[pred<1e-6] = 1e-6
    pred_log[pred>1-1e-6] = 1-1e-6
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds

    # if not visible or not labelled, ignore the corresponding joints loss
    num_pos  = pos_inds.float().sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1)
    neg_loss = neg_loss.sum(-1).sum(-1)
    mask = num_pos>0
    #loss[~mask] = loss[~mask] - neg_loss[~mask]
    # print(mask)
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / num_pos[mask]
    return loss.mean(-1)

def calc_losses(args, pred_cam,pred_3d_joints_from_token,pred_3d_vertices_coarse,pred_3d_vertices_fine,
                gt_3d_vertices_coarse,gt_3d_vertices_fine,gt_3d_joints_with_tag,
                gt_2d_joints,has_3d_joints,has_2d_joints,has_mesh,
                criterion_keypoints,criterion_2d_keypoints,criterion_vertices,
                mano_model,heatmap, heatmap1,criterion_heatmap,
                MAP, need_hloss=True):


    ######
    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)  # batch_size X 21 X 3
    pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, cfg.J_NAME.index('Wrist'), :]
    # normalize predicted vertices
    pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None,
                                                    :]  # batch_size X 778 X 3
    pred_3d_vertices_coarse = pred_3d_vertices_coarse - pred_3d_joints_from_mano_wrist[:, None,
                                                        :]  # batch_size X 195 X 3
    # normalize predicted joints
    pred_3d_joints_from_mano = pred_3d_joints_from_mano - pred_3d_joints_from_mano_wrist[:, None,
                                                          :]  # batch_size X 21 X 3
    pred_3d_joints_from_token_wrist = pred_3d_joints_from_token[:, 0, :]
    pred_3d_joints_from_token = pred_3d_joints_from_token - pred_3d_joints_from_token_wrist[:, None,
                                                            :]  # batch_size X 21 X 3
    # obtain 2d joints, which are projected from 3d joints of mano mesh
    pred_2d_joints_from_mano = orthographic_projection(pred_3d_joints_from_mano.contiguous(),
                                                       pred_cam.contiguous())  # batch_size X 21 X 2
    pred_2d_joints_from_token = orthographic_projection(pred_3d_joints_from_token.contiguous(),
                                                        pred_cam.contiguous())  # batch_size X 21 X 2

    # compute 3d joint loss
    loss_3d_joints = (keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_token, gt_3d_joints_with_tag,
                                       has_3d_joints, args.device) + \
                      keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_mano, gt_3d_joints_with_tag,
                                       has_3d_joints, args.device))
    # compute 3d vertex loss
    loss_3d_vertices = (
            args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_3d_vertices_coarse,
                                                             gt_3d_vertices_coarse, has_mesh, args.device) + \
            args.vloss_w_full * vertices_loss(criterion_vertices, pred_3d_vertices_fine,
                                                           gt_3d_vertices_fine, has_mesh, args.device))

    # compute 2d joint loss
    loss_2d_joints = (
            keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_token, gt_2d_joints, has_2d_joints) + \
            keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_mano, gt_2d_joints, has_2d_joints))


    heatmap_loss = 0

    if need_hloss:
        gt_2d_vertices = orthographic_projection(gt_3d_vertices_coarse, pred_cam)
        #
        # #gt_2d_vertices = orthographic_projection(gt_3d_joints_with_tag, pred_cam)
        gt, flag = make_gt(gt_2d_vertices,has_mesh, MAP)
        #
        heatmap_loss = calc_heatmap_loss(heatmap,gt,has_mesh,flag, criterion_heatmap)

        #新加上joint loss
        gt_2d_vertices1 = orthographic_projection(gt_3d_joints_with_tag, pred_cam)
        gt1, flag1 = make_gt(gt_2d_vertices1, has_mesh, MAP)
        heatmap_loss_joint = calc_heatmap_loss(heatmap1, gt1, has_mesh, flag1, criterion_heatmap)
        ##end

    mesh_htloss = args.heatmap_loss_weight * heatmap_loss
    jot_htloss = args.heatmap1_loss_weight * heatmap_loss_joint
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_3d_vertices\
           + args.vertices_loss_weight * loss_2d_joints + mesh_htloss + jot_htloss
           # + args.vertices_loss_weight * loss_surface

    return  pred_2d_joints_from_mano, pred_3d_vertices_fine, loss_2d_joints, loss_3d_joints, loss_3d_vertices, loss, heatmap_loss, heatmap_loss_joint