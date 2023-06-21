# coding=utf-8

import torch
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from utils.total_capture import amass_rnn as total
import model.RNN_6_model as nnmodel
from utils.opt import Options
from utils import loss_func, utils_utils as utils


def main(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    ignore = opt.ignore
    print(" torch.cuda.is_available()", torch.cuda.is_available())
    # device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjs = opt.adjs
    batch_size = opt.train_batch
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_rnn_pro_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")

    model = nnmodel.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                mid_size=18, output_size=72, adjs=adjs,
                                batch_size=batch_size, device=device, dropout=0.2)
    if is_cuda:
        model = model.to(device)

    model_path_len = "/home/xt/Skim-IMU-master/checkpoint/test" \
                     "/ckpt_main_rnn_gcn_progcn_pro_new_5_in36_out24_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_path_len))

    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')

    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    # data loading
    print(">>> loading data")
    test_dataset = total(path_to_data=opt.data_amass_xt_dir, input_n=input_n, output_n=output_n,
                         split=0)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 128
        shuffle=False,
        num_workers=opt.job,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    ret_log = np.array([1])
    head = np.array(['epoch'])
    test_position, test_bone = test(train_loader=test_loader, model=model, cuda=device,
                                    ignore=ignore, batch_size=batch_size,
                                    input_n=input_n, all_n=all_n)
    position = 'po_'
    bone = 'bo_'
    # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
    ret_log = np.append(ret_log, [test_position, test_bone])
    head = np.append(head,
                     [position + '5', position + '11', position + '17', position + '23',
                      position + '29', position + '35',
                      position + '38', position + '41', position + '454', position + '47', position + '50',
                      position + '53', position + '56', position + '59'
                      ])
    head = np.append(head,
                     [bone + '5', bone + '11', bone + '17', bone + '23', bone + '29', bone + '35',
                      bone + '38', bone + '41', bone + '454', bone + '47', bone + '50',
                      bone + '53', bone + '56', bone + '59'
                      ])
    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。

    df.to_csv(opt.ckpt + 'eval'+'/' + script_name + '.csv', header=head, index=False)


def test(train_loader, model, cuda, ignore, batch_size, input_n, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    # eval_frame = [2, 5, 8, 11, 14, 29]
    # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
    eval_frame = [5, 11, 17, 23, 29, 35, 38, 41, 44, 47, 50, 53, 56, 59]

    t_posi = np.zeros(len(eval_frame))  # 6位
    t_posi_yi = np.zeros(len(eval_frame))
    model.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            input_batch_size = input_ori.shape[0]  # 16
            if input_batch_size != batch_size:
                continue
            if torch.cuda.is_available():
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_joints = out_joints.to(cuda).float()
            # model要改
            y_out = model(input_ori, input_acc)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_joints = y_out[:, j, :], out_joints[:, j, :]
                t_posi[k] += loss_func.position_loss(test_out, test_joints).cpu().data.numpy() * batch * 100

                t_posi_yi[k] += loss_func.position_loss_yi(test_out, test_joints,
                                                           ignore).cpu().data.numpy() * batch * 100
        N += batch_size

    return t_posi / N, t_posi_yi / N


# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  # 矩阵求逆
    return dct_m, idct_m


def get_dct(out_joints, input_n):
    batch, frame, dim = out_joints.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    pad_idx = np.repeat([input_n - 1], frame - input_n)  # 25个(10-1)
    i_idx = np.append(np.arange(0, input_n), pad_idx)  # 前十个是输入，后二十五个是最后一个姿势
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints[i_idx, :])
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.transpose(1, 0).reshape(batch, dim, frame)
    return input_joints


def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape

    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).to(torch.float32).to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    # 50,32*24*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
    # 32,72,50
    pred_3d = outputs_p3d.contiguous().view(-1, 3).contiguous()
    targ_3d = out_joints.contiguous().view(-1, 3).contiguous()
    return pred_3d, targ_3d


def test_gcn(train_loader, model, cuda, ignore, batch_size, input_n, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    # eval_frame = [2, 5, 8, 11, 14, 29]
    # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
    eval_frame = [5, 11, 17, 23, 29, 35, 38, 41, 44, 47, 50, 53, 56, 59]

    t_posi = np.zeros(len(eval_frame))  # 6位
    t_posi_yi = np.zeros(len(eval_frame))
    model.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        batch_size = input_acc.shape[0]  # 16
        if batch_size == 1:
            continue
        # out_joints:(32,24,3,frames)
        input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
        model_input = get_dct(input_ori_acc, input_n)
        if torch.cuda.is_available():
            # print("x")
            model_input = model_input.to(cuda).float()
            out_joints = out_joints.to(cuda).float()
        y_out = model(model_input)  # 32,72,50
        batch, frame, node, dim = out_joints.data.shape

        _, idct_m = get_dct_matrix(frame)
        idct_m = torch.from_numpy(idct_m).to(torch.float32).to(cuda)
        outputs_t = y_out.view(-1, frame).permute(1, 0)
        # 50,32*24*3
        outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
        outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            test_out, test_joints = outputs_p3d[:, j], out_joints[:, j]
            t_posi[k] += loss_func.position_loss(test_out, test_joints).cpu().data.numpy() * batch * 100

            t_posi_yi[k] += loss_func.position_loss_yi(test_out, test_joints,
                                                       ignore).cpu().data.numpy() * batch * 100
        N += batch_size

    return t_posi / N, t_posi_yi / N


if __name__ == "__main__":
    option = Options().parse()
    main(option)
