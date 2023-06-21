# coding=utf-8

import torch

import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

from utils.amass_6_node import amass_rnn as amass
import model.RNN_best as nnmodel
from utils.opt import Options
from utils import loss_func, utils_utils as utils
import utils.viz as viz


def main(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = opt.all_n

    print(" torch.cuda.is_available()", torch.cuda.is_available())
    # device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjs = opt.atts
    batch_size = opt.train_batch
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    # in_features=all_n, adjs=adjs, dim=3, cuda=device, p_dropout=0.5
    model = nnmodel.Predict_imu(input_frame_len=input_n, output_frame_len=output_n, input_size=120, output_size=30,
                                batch_size=batch_size, adjs=adjs, num_layers=[2, 1], device=device, dropout=0.1)
    if is_cuda:
        model = model.to(device)

    model_path_len = './checkpoint/test/ckpt_main_10_rnn_posi+accrnn_2000_in30_out20_dctn50_best.pth.tar'
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
    test_dataset = amass(path_to_data=opt.data_benji_dir, input_n=input_n, output_n=output_n,
                         split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 128
        shuffle=False,
        num_workers=opt.job,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if batch_size != 32:
            continue
        if torch.cuda.is_available():
            input_ori = input_ori.to(device).float()
            input_acc = input_acc.to(device).float()
            out_poses = out_poses.to(device).float()
            out_joints = out_joints.to(device).float()
        y_out = model(input_ori, input_acc)
        # b1_out, b2_out, b3_out, b4_out, y_out = model(input_acc, input_ori)
        # 32,50,24,3
        batch, frame, _ = y_out.data.shape
        xyz_gt = out_joints.cpu().data.numpy()
        xyz_pred = y_out.cpu().data.numpy()
        # 第一位作为batch
        for k in range(8):
            plt.cla()  # 清除当前轴
            figure_title = "seq:{},".format((k + 1))
            viz.plot_predictions(xyz_gt[k, :, :], xyz_pred[k, :, :], fig, ax, figure_title)
            plt.pause(0.5)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
