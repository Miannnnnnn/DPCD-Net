import argparse
import os
import torch
import copy
import sys

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from attrdict import AttrDict

from dpcnet.data.loader import data_loader
# from dpcnet.models import TrajectoryGenerator
# from dpcnet.models_prior_unet_burgers import TrajectoryGenerator
from dpcnet.models_prior_unet_burgers import TrajectoryGenerator
# from dpcnet.models_gph_simvp import TrajectoryGenerator
from dpcnet.losses import displacement_error, final_displacement_error,toNE,trajectory_displacement_error,value_error,trajectory_diff,value_diff
from dpcnet.utils import relative_to_abs, get_dset_path,dic2cuda

from dpcnet.merge_net_all import Merge_Net_All

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default=r'model_save/models', type=str)
parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
    )
    #generator.load_state_dict(checkpoint['g_state'])
    generator.load_state_dict(checkpoint['g_state'], strict=False)
    generator.cuda()
    generator.eval()
    return generator


def getmin_helper(error,an,timeanpv, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    an = torch.stack(an, dim=1)
    timeanpv = torch.stack(timeanpv, dim=1)
    minpoint,minpoint_pv = [],[]


    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.argmin(_error)
        # _error = torch.mean(_error)
        minpoint.append(an[start:end,_error.data.cpu()])
        minpoint_pv.append(timeanpv[start:end,_error.data.cpu()])

    # minpoint = torch.stack(minpoint,dim=1).squeeze()
    # minpoint_pv = torch.stack(minpoint_pv,dim=1).squeeze()
    minpoint = torch.stack(minpoint,dim=1).squeeze(0)
    minpoint_pv = torch.stack(minpoint_pv,dim=1).squeeze(0)
    return {'tr':minpoint,'pv':minpoint_pv}

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        # _error = torch.mean(_error)

        sum_ += _error
    return sum_

def ve_evaluate_helper(error, seq_start_end):
    sum_p = 0
    sum_v = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error,dim=0)
        # _error = _error[0]
        # _error = torch.mean(_error,dim=0)
        sum_p += _error[0][0]
        sum_v += _error[0][1]
    return sum_p,sum_v

def evaluate(args, loader, generator, num_samples,sava_path):
    ade_outer, fde_outer,tde_outer,ve_outer,ana_outer,pv_outer,gt = [], [],[],[],[],[],[]
    total_traj = 0
    with torch.no_grad():
        for batch in loader:

            # #env_data = dic2cuda(batch[-2])
            # #batch = [tensor.cuda() for tensor in batch[:-2]]
            batch = [tensor.cuda() for tensor in batch[:-1]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
             obs_date_mask, pred_date_mask,
             u_image_obs, v_image_obs, u_image_pre, v_image_pre,
             image_obs,image_pre) = batch

            ade, fde,tde,ve = [], [],[],[]
            analyse,analyse_pv = [],[]
            total_traj += pred_traj_gt.size(1)
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            gt.append(torch.cat([pred_traj_gt.permute(1, 0, 2), pred_traj_gt_Me.permute(1, 0, 2)], dim=2))
            # pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)


            # pred_traj_fake_rel,_,_,_ = generator(
            #     obs_traj, obs_traj_rel, seq_start_end,image_obs,env_data,
            #     num_samples=num_samples, all_g_out=False)
            # pred_traj_fake_rel, _, _, _,_,_ = generator(
            #     obs_traj, obs_traj_rel, seq_start_end, image_obs,
            #     u_image_obs, v_image_obs,
            #     num_samples=num_samples, all_g_out=False)
            pred_traj_fake_rel, _, _, _, _ = generator(
                obs_traj, obs_traj_rel, seq_start_end, image_obs,
                u_image_obs, v_image_obs,
                num_samples=num_samples, all_g_out=False)
            pred_traj_fake_relt = pred_traj_fake_rel
            pred_traj_fake_rel = pred_traj_fake_relt[:,:,:,:2]
            pred_traj_fake_rel_Me = pred_traj_fake_relt[:,:,:,2:]
            #print(pred_traj_fake_rel.shape)
            # pred_traj_fake_rel 用来预测后12个点与第8点的偏差
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1,:,:2]
            )
            pred_traj_fake_rel_Me = relative_to_abs(
                pred_traj_fake_rel_Me, obs_traj_Me[-1]
            )
            # 只看坐标的偏差

            # 函数会改变参数变量
            real_pred_traj_gt,real_pred_traj_gt_Me = toNE(copy.deepcopy(pred_traj_gt),copy.deepcopy(pred_traj_gt_Me))

                # ade.append(displacement_error(
                #     pred_traj_fake, pred_traj_gt, mode='raw'
                # ))
            for sample_i in range(num_samples):
                real_pred_traj_fake, real_pred_traj_fake_Me = toNE(pred_traj_fake[:,sample_i].squeeze(0),
                                                                   pred_traj_fake_rel_Me[:,sample_i].squeeze(0))
                analyse.append(trajectory_diff(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                analyse_pv.append(value_diff(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))
                tde.append(trajectory_displacement_error(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                ve.append(value_error(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))

                # fde.append(final_displacement_error(
                #     pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                # ))
            time_tde_sum = []
            time_ve_sum = []
            time_an_sum = []
            time_anpv_sum = []

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                timean = [x[:, i] for x in analyse]
                timeanpv = [x[:, i] for x in analyse_pv]
                out = getmin_helper(timeade,timean,timeanpv, seq_start_end)
                time_an_sum.append(out['tr'])
                time_anpv_sum.append(out['pv'])
            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                time_tde_sum.append(evaluate_helper(timeade, seq_start_end))

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in ve]
                time_ve_sum.append(ve_evaluate_helper(timeade, seq_start_end))
            # ade_sum = evaluate_helper(ade, seq_start_end)
            # fde_sum = evaluate_helper(fde, seq_start_end)

            tde_outer.append(time_tde_sum)
            ve_outer.append(time_ve_sum)

            time_an_sum = torch.stack(time_an_sum,dim=1)
            ana_outer.append(time_an_sum)
            time_anpv_sum = torch.stack(time_anpv_sum, dim=1)
            pv_outer.append(time_anpv_sum)
            # saveana(ana_outer, pv_outer, gt, sava_path)
            # fde_outer.append(fde_sum)
       
        saveana(ana_outer, pv_outer, gt, sava_path)
        tde_outer = torch.tensor(tde_outer)
        ve_outer = torch.tensor(ve_outer)
        ade = torch.sum(tde_outer,dim=0) / (total_traj)
        ve = torch.sum(ve_outer,dim=0) / (total_traj)
        fde = 0
        return ade, ve

import numpy as np
def saveana(ana_outer,ve_outer,gt,sava_path):
    ana_outer = torch.cat(ana_outer, dim=0)
    ana_outer_np = ana_outer.data.cpu().numpy()
    ve_outer = torch.cat(ve_outer, dim=0)
    ve_outer_np = ve_outer.data.cpu().numpy()
    gt = torch.cat(gt, dim=0)
    gt_np = gt.data.cpu().numpy()
    traj_path = os.path.join(sava_path, 'trajectory.npy')
    pv_path = os.path.join(sava_path, 'pvdif.npy')
    gt_path = os.path.join(sava_path, 'gt.npy')
    if os.path.exists(traj_path):
        tra = np.load(traj_path)
        np.save(traj_path,(ana_outer_np+tra)/2)
    else:
        np.save(traj_path, ana_outer_np)
    if os.path.exists(pv_path):
        pv = np.load(pv_path)
        np.save(pv_path, (ve_outer_np+pv)/2)
    else:
        np.save(pv_path, ve_outer_np)
    np.save(gt_path, gt_np)


def print_log(modelpath,tde,ve,_args,f_path,mode):
    f = open(f_path,mode)
    print(os.path.split(modelpath)[1])
    print('Dataset: {}, Pred Len: {}'.format(
        _args.dataset_name, _args.pred_len))
    print('TDR:', tde)
    print('TDR:', ve)
    print(os.path.split(modelpath)[1],file=f)
    print('Dataset: {}, Pred Len: {}'.format(
        _args.dataset_name, _args.pred_len),file=f)
    print('TDR:', tde,file=f)
    print('TDR:', ve,file=f)
    f.close()

def main(args):
    sava_path = args.model_path
    log_file = os.path.join(sava_path,'result_log.txt')
    best_file = os.path.join(sava_path,'result_best.txt')
    min_error = 10000000
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:


         if 'no_' in path or 'pt' not in path :
              continue
        modelpath = path
        checkpoint = torch.load(path)
        # print(checkpoint['args'])
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        tde, ve = evaluate(_args, loader, generator, args.num_samples,sava_path)
        print_log(modelpath,tde,ve,_args,log_file,'a')
        if torch.mean(tde).item() < min_error:
            min_error =torch.mean(tde).item()
            print('now_best==================:')
            print_log(modelpath, tde, ve, _args, best_file,'w')



    return tde,ve


def seed_torch():

    seed = 1024
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    num = 1
    for i in range(num):
        ted,ve = main(args)
        if i == 0:
            ated, ave = ted,ve
        else:
            ated += ted
            ave+=ve
    print(ated/num,ave/num)

