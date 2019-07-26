import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import ssim_module
import numpy as np
import torch.nn as nn
import warnings
import numbers

if hasattr(F, 'interpolate'):
    interpolate = torch.nn.functional.interpolate 
else:
    from .interpolate_func import  interpolate


def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size

def EPE_with_mask(input_flow, target_flow, mask, sparse=False, mean=True):
    if mask is not None:
        target_flow = target_flow * mask
        input_flow = input_flow * mask
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size

def sparse_max_pool(input, size):
    """Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points."""

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def PyramidalLoss(network_output, target_flow, mask_pyramid=None, weights=None, sparse=False, pred_on_target=True, mask_levels=[]):
    """
    if pred_on_target=True, pred the optical flow directly on target scale (don not scale against on each level)
    """
    def one_scale(output, target, mask, sparse):
        b, _, h, w = output.size()
        if mask is not None:
            mb,_, mh, mw = mask.shape
            assert [mh, mw] == [h, w]
        bt, _, ht, wt = target.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = interpolate(target, (h, w), mode='area')

        if not pred_on_target:
            # estimated flow at each level has real displacements
            target_scaled[:,0,:,:] = target_scaled[:,0,:,:] * (w/wt)
            target_scaled[:,1,:,:] = target_scaled[:,1,:,:] * (h/ht)

        return EPE_with_mask(output, target_scaled, mask,  sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    
    loss = 0
    for i, (output, weight) in enumerate(zip(network_output, weights)):
        if mask_pyramid is not None  and (i in mask_levels):
            mask = mask_pyramid[i]
        else:
            mask = None
        loss += weight * one_scale(output, target_flow, mask, sparse)
        if mask is not None:
            loss += 1e-2 * weight *(1.0-mask).sum() 
    return loss


def realEPE(args, output, target):
    """
    sparse: use for KITTI dataset
    div_flow: whether multipy by div_flow factor
    pred_on_target: If True, the predicted flow(output) has the same scale as on the target level
                    otherwise, it predicted the flow on its currect level. It depends on whether 
                    rescaling the gt flow on each level when training.
    compute_size: If None, compute on the target size, otherwise, rescale both the target and the
                    output to the compute_size first, then compute the EPE on the rescaled size.
                    (use to compute the actual EPE for benchmarking.)
    """
    sparse = args.sparse
    div_flow = args.div_flow
    pred_on_target = not args.rescale_each_level
    compute_size = args.compute_size

    if div_flow != 1.0:
        target = target * div_flow
        output = output * div_flow

    b, _, th, tw = target.size()
    ob, _, oh, ow = output.size()
    ch, cw = [th, tw] if compute_size is None else compute_size

    if [th, tw] != [ch, cw]:
        target =  interpolate(target, (ch, cw), mode='bilinear', align_corners=False)
        target[:,0,:,:] = target[:,0,:,:] * (cw/tw)
        target[:,1,:,:] = target[:,1,:,:] *  (ch/th)
    
    if [oh, ow] != [ch, cw]:
        output = interpolate(output, (ch, cw), mode='bilinear', align_corners=False) 
    
    if pred_on_target:
        output[:,0,:,:] = output[:,0,:,:] * (cw/tw)
        output[:,1,:,:] = output[:,1,:,:] *  (ch/th)
    else:
        output[:,0,:,:] = output[:,0,:,:] * (cw/ow)
        output[:,1,:,:] = output[:,1,:,:] *  (ch/oh)
   
    return EPE(output, target, sparse, mean=True)



# --------------------------- Reconstruct Loss -------------------------- #
grid_cache = {}
def warp(x, flo):
    B, C, H, W = x.size()
    grid_name = "{}_{}_{}".format(B,H,W)
    if grid_name not in grid_cache.keys():
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        grid = grid.cuda()
        vgrid = Variable(grid) + flo

        vgl = [2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0, 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0]
        vgrid = torch.stack(vgl, 1)
        vgrid = vgrid.permute(0,2,3,1)        
        grid_cache[grid_name] = vgrid
    vgrid = grid_cache[grid_name]

    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    grid = grid.cuda()
    vgrid = Variable(grid) + flo

    vgl = [2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0, 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0]
    vgrid = torch.stack(vgl, 1)
    vgrid = vgrid.permute(0,2,3,1)        
    
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    x_warped = output*mask 
    return x_warped


def multiscale_reconstruct_loss(args, frames, flow_pyramid, mask_pyramid=None, compute_levels=[0, 1, 2], fwd=True):
    """
    Compute multi-scale reconstruct loss
    :param div_flow:
    :param compute_levels:
    :param frames:
    :param flow_pyramid: list(batch_size, K, snippet_len-1, C, H, W)
    :param compute_size: rescale to  [compute_size * compute_size] to compute the loss
    :return:
    """
    sparse = args.sparse
    div_flow = args.div_flow
    mask_levels = args.mask_levels 
    compute_size = args.compute_size 

    warnings.warn("sparse option in multiscale_reconstruct_loss() function is not supported yet!")   
    snippet_len, f_c, f_h, f_w = frames.size()[-4:]
    if compute_size is not None:
        w_h, w_w = compute_size
    
    if frames.dim() > 5:
        frames = frames.view((-1,) + frames.size()[-4:])
    assert fwd, " backword loss is not supported again!"
    
    def charbonnier_penalty(x, e=1e-8, delta=0.4, averge=True):
        p = ((x)**2 + e ).pow(delta)
        if averge:
            p=p.mean()
        else:
            p=p.sum()
        return p

    def reconstruct_loss(f1, f2, averge=True):
        beta_1, beta_2, beta_3 = 0.5, 0.6, 0.5
        diff_loss = charbonnier_penalty(f2 - f1, delta=0.4, averge=True)
        ssim_loss = 1.0 - ssim_module.ssim(f1, f2, window_size=11, size_average=True)  # [0, 1]
        # psnr_loss = -10.0 * ((1.0 / (square_.mean() + 1)).log10())
        psnr_loss = 0.0

        num = 1 if averge else f1.shape[-2] * f1.shape[-1] 
        diff_loss = num *  beta_1 * diff_loss 
        ssim_loss = num * beta_2 * ssim_loss 
        psnr_loss = num * beta_3 * psnr_loss 

        return [diff_loss, ssim_loss, psnr_loss]

    
    # weights = [0.4, 0.3, 0.2, 0.1, 0.1]
    weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    smoothness_weight = 2e-3
    mask_weight = 1e-2
    
    mask_loss, smooth_loss, diff_loss, ssim_loss, psnr_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    for snp in  range(0, snippet_len-1):
        frame_1 = frames[:,snp,:,:,:]
        frame_2 = frames[:,snp+1,:,:,:]

        for i, flow in enumerate(flow_pyramid): 
            if i not in compute_levels:
                continue
            
            # flo = flow.view((-1,) + flow.size()[-3:])
            flo = flow[:,0,snp,:,:,:]
            if div_flow != 1.0:
                flo = flo * div_flow
            if compute_size is None:
                if list(frame_1.shape[-2:]) != list(flo.shape[-2:]):
                    frame_1 = interpolate(frame_1, flo.shape[-2:], mode='bilinear', align_corners=False)  
                    frame_2 = interpolate(frame_2, flo.shape[-2:], mode='bilinear', align_corners=False)  
            else:
                f_h, f_w = frame_1.shape[-2:]
                if list(frame_1.shape[-2:]) != [w_h, w_w]:
                    frame_1 = interpolate(frame_1, (w_h, w_w), mode='bilinear', align_corners=False)
                    frame_2 = interpolate(frame_2, (w_h, w_w), mode='bilinear', align_corners=False)
                if list(flo.shape[-2:]) != [w_h, w_w]:
                    flo = interpolate(flo, (w_h, w_w), mode='bilinear', align_corners=False)
                    flo[:, 0, :, :] *= (w_w / f_w)
                    flo[:, 1, :, :] *= (w_h / f_h)

            warped_frames = warp(frame_2, flo)
            if mask_pyramid is not None and (i in mask_levels):
                mask = mask_pyramid[i]
                frame_1  = frame_1 * mask
                warped_frames = warped_frames  *mask
                _mask_loss =  (1.0 - mask).sum()
            else:
                _mask_loss = 0.0

            _diff_loss, _ssim_loss, _psnr_loss = reconstruct_loss(frame_1, warped_frames, averge=False)

            diff_loss += weights[i] * _diff_loss
            ssim_loss += weights[i] * _ssim_loss
            psnr_loss += weights[i] * _psnr_loss

            # smoothness_loss
            #d_uv_x = charbonnier_penalty(flo[:, :, :, 2:] + flo[:, :, :, :-2] -  2*flo[:, :, :, 1:-1], averge=False)
            #d_uv_y = charbonnier_penalty(flo[:, :, 2:, :] + flo[:, :, :-2, :] -  2*flo[:, :, 1:-1, :], averge=False)
            d_uv_x = charbonnier_penalty(flo[:, :, :, 1:] - flo[:, :, :, :-1], averge=False)
            d_uv_y = charbonnier_penalty(flo[:, :, 1:, :] - flo[:, :, :-1, :], averge=False)
            smooth_loss +=  smoothness_weight * weights[i] * (d_uv_x +  d_uv_y)
            mask_loss +=    mask_weight * weights[i] *  _mask_loss

    if args.print_loss:
        print("diff: %.3f ssim: %.3f psnr: %.3f, mask: %.3f, smooth: %.3f " %(
            diff_loss if isinstance(diff_loss, numbers.Number) else diff_loss.item(),
            ssim_loss if isinstance(ssim_loss, numbers.Number) else ssim_loss.item(), 
            psnr_loss if isinstance(psnr_loss, numbers.Number) else psnr_loss.item(),
            mask_loss if isinstance(mask_loss, numbers.Number) else mask_loss.item(), 
            smooth_loss if isinstance(smooth_loss, numbers.Number) else smooth_loss.item()))

    all_losses =  mask_loss + smooth_loss + diff_loss + ssim_loss + psnr_loss
    return all_losses

