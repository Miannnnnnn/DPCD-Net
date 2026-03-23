import logging
import os
import math

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader

def seq_collate(data):
  
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask,
     u_image_obs, u_image_pre, v_image_obs, v_image_pre,
     image_obs, image_pre, tyID) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    obs_traj_Me = torch.cat(obs_traj_Me, dim=0).permute(2, 0, 1)
    pred_traj_Me = torch.cat(pred_traj_gt_Me, dim=0).permute(2, 0, 1)
    obs_traj_rel_Me = torch.cat(obs_traj_rel_Me, dim=0).permute(2, 0, 1)
    pred_traj_rel_Me = torch.cat(pred_traj_gt_rel_Me, dim=0).permute(2, 0, 1)
    obs_date_mask = torch.cat(obs_date_mask, dim=0).permute(2, 0, 1)
    pred_date_mask = torch.cat(pred_date_mask, dim=0).permute(2, 0, 1)

    u_image_obs = torch.stack(u_image_obs, dim=0).permute(0, 4, 1, 2, 3)
    u_image_pre = torch.stack(u_image_pre, dim=0).permute(0, 4, 1, 2, 3)
    v_image_obs = torch.stack(v_image_obs, dim=0).permute(0, 4, 1, 2, 3)
    v_image_pre = torch.stack(v_image_pre, dim=0).permute(0, 4, 1, 2, 3)

    image_obs = torch.stack(image_obs, dim=0).permute(0,4,1,2,3)  # 转换后 image shape:(batch,channel,seq_len,w,h)
    image_pre = torch.stack(image_pre, dim=0).permute(0,4,1,2,3)
  
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
        obs_date_mask, pred_date_mask,
        u_image_obs, u_image_pre, v_image_obs, v_image_pre,
        image_obs, image_pre, tyID
    ]

    return tuple(out)



def read_file(_path, delim='\t'):
    data = []
    add = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            add.append(line[-2:])
            line = [float(i) for i in line[:-2]]
            data.append(line)
    return {'main':np.asarray(data),'addition':add}


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=4, skip=1, threshold=0.002,
        min_ped=1, delim='\t',other_modal='gph'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.modal_name = other_modal

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_date_mask = []
        loss_mask_list = []
        non_linear_ped = []
        tyID = []

        # 迭代读取文件夹中的txt文件
        for path in all_files:
            _,x = os.path.split(path)
            tyname = os.path.splitext(x)[0]
            data = read_file(path, delim)
            addinf = data['addition']
            data = data['main']
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                # 将txt文件中同一帧的，没有目标的坐标点保存在同一个frame_data的同一个index中
                frame_data.append(data[frame == data[:, 0], :])
            # 获取一个txt文件中的轨迹可以分割成多少个子轨迹，如第一个子轨迹：1,2,3...20，第二2,3,4，...21
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            # 迭代子轨迹
            for idx in range(0, num_sequences * self.skip + 1, skip):
                # axis=0  照着row的方向叠在一起
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                # peds_in_curr_seq  存储当前子序列的目标id，   ty的话   应该就是1
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                # 获取时间信息
                curr_date_mask = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    ##
                    curr_ped_date_mask = [x[0] for x in addinf[idx:idx+pred_len+obs_len]]
                    curr_ped_date_mask = self.embed_time(curr_ped_date_mask)
                    # Make coordinates relative
                    # rel_curr_ped_seq 存相邻两个坐标点中间的差  后-前
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_date_mask[_idx, :, pad_front:pad_end] = curr_ped_date_mask
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_date_mask.append(curr_date_mask[:num_peds_considered])
                    tyID.append({'old':[tyname,idx],'new':addinf[idx+self.obs_len-1],
                                 'tydate':[x[0] for x in addinf[idx:idx+pred_len+obs_len]]})

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_date_mask = np.concatenate(seq_list_date_mask, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_Me = torch.from_numpy(
            seq_list[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_Me = torch.from_numpy(
            seq_list[:, 2:, self.obs_len:]).type(torch.float)
        self.obs_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        self.obs_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, :self.obs_len]).type(torch.float)
        self.pred_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, self.obs_len:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # seq_start_end 一个元组的索引代表的是同子序列中不同id的轨迹
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.tyID = tyID

    def __len__(self):
        return self.num_seq

    def embed_time(self,date_list):
        data_embed = []
        for date in date_list:
            year = (float(date[:4]) - 1949) / (2019 - 1949) - 0.5
            month = (float(date[4:6]) - 1) / 11.0 - 0.5
            day = (float(date[6:8]) - 1) / 30.0 - 0.5
            hour = float(date[8:10]) / 18 - 0.5
            data_embed.append([year, month, day, hour])
        return np.array(data_embed).transpose(1, 0)[np.newaxis, :, :]

    def transforms_gph(self,img):
        modal_range = {'gph': (44490.578125,58768.4486860389),
                       '10v': (-48.2635498046875, 53.091079711914055),
                       '10u': (-49.08894348144531, 47.56512451171875)
                      }

        all_min,all_max = modal_range[self.modal_name]
        img = (img-all_min)/(all_max-all_min)
        img[img>1] = 1
        img[img<0] = 0
        return img


    def transforms_u(self,img):
        modal_range = {'10u': (-42.69940185546875, 42.42747497558593)}

        all_min,all_max = modal_range['10u']
        denominator = all_max - all_min
        if abs(denominator) < 1e-5:
            denominator = 1e-5  # 防止除以零
        img = (img - all_min) / denominator
        img = np.clip(img, 0, 1)
        return img


    def transforms_v(self,img):
        modal_range = {'10v': (-39.61431884765626, 41.31439208984375)}

        all_min,all_max = modal_range['10v']
        denominator = all_max - all_min
        if abs(denominator) < 1e-5:
            denominator = 1e-5  # 防止除以零
        img = (img - all_min) / denominator
        img = np.clip(img, 0, 1)
        return img

    def img_read_gph(self,img_path):
        # print(img_path)

        # img = cv2.imread(img_path)
        # img = cv2.resize(img,(128,64))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.load(img_path)

        img = cv2.resize(img,(64,64))
        img = self.transforms_gph(img)
        img = img[:,:,np.newaxis]
        return img

    def img_read_u(self, img_path):
        img = np.load(img_path)
        img = cv2.resize(img, (64, 64))
        img = self.transforms_u(img)
        img = img[:, :, np.newaxis]
        return img

    def img_read_v(self,img_path):
        # print(img_path)

        # img = cv2.imread(img_path)
        # img = cv2.resize(img,(128,64))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.load(img_path)
        img = cv2.resize(img, (64, 64))
        img = self.transforms_v(img)
        img = img[:, :, np.newaxis]
        return img



    def get_img_gph(self,tyid_dic):
        tyname = tyid_dic['new'][1]
        year = tyid_dic['old'][0][2:6]
        tydate = tyid_dic['tydate']

        modal_path = {'gph':r'D:\experiment\DPCD-Net\datasets\geopotentialp',
                      }
        data_dir = os.path.join(modal_path[self.modal_name], year, tyname)
        image_obs = []
        image_pre = []
        obs_list = tydate[:self.obs_len]
        pre_list = tydate[self.obs_len:]
        for obs_date in obs_list:
            img_path = os.path.join(data_dir, obs_date + '.npy')
            img = self.img_read_gph(img_path)
            image_obs.append(img)
        for pre_date in pre_list:
            img_path = os.path.join(data_dir, pre_date + '.npy')
            img = self.img_read_gph(img_path)
            image_pre.append(img)
        image_obs = torch.tensor(np.array(image_obs), dtype=torch.float)
        image_pre = torch.tensor(np.array(image_pre), dtype=torch.float)
        # print('image_obs:',image_obs)
        # print('image_pre:',image_pre)
        # return {'obs': image_obs, 'pre': image_pre,'env':env_data}
        return {'obs': image_obs, 'pre': image_pre}

    def get_img_u(self,tyid_dic):
        tyname = tyid_dic['new'][1]
        year = tyid_dic['old'][0][2:6]
        tydate = tyid_dic['tydate']

        data_dir_u = os.path.join(r'D:\experiment\uv_data\u', year, tyname)

        image_obs_u = []
        image_pre_u = []
        obs_list = tydate[:self.obs_len]
        pre_list = tydate[self.obs_len:]
        for obs_date in obs_list:
            img_path_u = os.path.join(data_dir_u, obs_date + '.npy')
            u_img = self.img_read_u(img_path_u)
            image_obs_u.append(u_img)
        for pre_date in pre_list:
            img_path_u = os.path.join(data_dir_u, pre_date + '.npy')
            u_img = self.img_read_u(img_path_u)
            image_pre_u.append(u_img)
        image_obs_u = torch.tensor(np.array(image_obs_u), dtype=torch.float)
        image_pre_u = torch.tensor(np.array(image_pre_u), dtype=torch.float)

      
        return {'u_obs': image_obs_u, 'u_pre': image_pre_u}
        
    def get_img_v(self,tyid_dic):
        tyname = tyid_dic['new'][1]
        year = tyid_dic['old'][0][2:6]
        tydate = tyid_dic['tydate']

        data_dir_v = os.path.join(r'D:\experiment\uv_data\v', year, tyname)
       
        image_obs_v = []
        image_pre_v = []
        obs_list = tydate[:self.obs_len]
        pre_list = tydate[self.obs_len:]
        for obs_date in obs_list:
            img_path_v = os.path.join(data_dir_v, obs_date + '.npy')
            v_img = self.img_read_v(img_path_v)
            image_obs_v.append(v_img)
        for pre_date in pre_list:
            img_path_v = os.path.join(data_dir_v, pre_date + '.npy')
            v_img = self.img_read_v(img_path_v)
            image_pre_v.append(v_img)
        image_obs_v = torch.tensor(np.array(image_obs_v), dtype=torch.float)
        image_pre_v = torch.tensor(np.array(image_pre_v), dtype=torch.float)

        return {'v_obs': image_obs_v, 'v_pre': image_pre_v}

    def __getitem__(self, index):
        # 迭代一次获取一个子序列的所有行人的轨迹，台风的话只有一个轨迹.
        # 这里的子轨迹只有20点，8或12作为先决条件，这样的话，就有一个假设就是：
        # 当前时刻的    将来轨迹只与近8或者12个点信息有关
        start, end = self.seq_start_end[index]
        image = self.get_img_gph(self.tyID[start:end][0])
        image_u = self.get_img_u(self.tyID[start:end][0])
        image_v = self.get_img_v(self.tyID[start:end][0])
        
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.obs_traj_Me[start:end, :], self.pred_traj_Me[start:end, :],
            self.obs_traj_rel_Me[start:end, :], self.pred_traj_rel_Me[start:end, :],
            self.obs_date_mask[start:end, :], self.pred_date_mask[start:end, :],
            image_u['u_obs'], image_v['v_obs'], image_u['u_pre'], image_v['v_pre'],
            image['obs'], image['pre'],
            self.tyID[start:end]
        ]

        return out


if __name__ == '__main__':
    path = r'D:\experiment\DPCD-Net\datasets\1950_2019\test'
    dset = TrajectoryDataset(path,obs_len=8,pred_len=4,skip=1,delim='\t')
    loader = DataLoader(dset,batch_size=16,shuffle=True,num_workers=4,collate_fn=seq_collate)
 
    for batch in loader:

        u_image_obs = batch[-7]
        v_image_obs = batch[-6]
        u_image_pre = batch[-5]
        v_image_pre = batch[-4]
        image_obs = batch[-3]
        image_pre = batch[-2]
  
