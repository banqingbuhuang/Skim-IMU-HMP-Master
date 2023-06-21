# coding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from utils.load_total import amass_rnn as total
from utils.amass_6_node import amass_rnn as amass
# import model.Predict_imu as nnmodel
import model.GCN as nnmodel
from utils.opt import Options
from utils import utils_utils as utils
from utils import loss_func

actions = ['acting', 'rom', 'walking', 'freestyle']


def main(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = 0.00001
    ckpt = opt.ckpt + '_GCN'
    ignore = opt.ignore
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:测试total_capture
    script_name += "nnnnn_n{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.GCN(input_feature=all_n, hidden_feature=128, p_dropout=0.5,
                        num_stage=12, node_n=72)

    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    train_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                          split=0)
    val_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                        split=1)
    test_dip_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                             split=2)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=1,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1024,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)

    test_dip_loader = DataLoader(
        dataset=test_dip_dataset,
        batch_size=1024,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)
    test_loader = dict()
    for act in actions:
        test_dataset = total(path_to_data=opt.data_xt_total_dir, act=act, input_n=input_n, output_n=output_n,
                             split=2)
        test_loader[act] = DataLoader(
            dataset=test_dataset,
            batch_size=1024,  # 128
            shuffle=False,
            num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
            pin_memory=False)
        print(">>> test data {}".format(test_loader[act].__len__()))
    print(">>> train data {}".format(train_dataset.__len__()))  # 32178
    print(">>> validation data {}".format(val_dataset.__len__()))  # 1271
    print(">>> test data {}".format(test_dip_loader.__len__()))
    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print('=====================================')
        print('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        Ir_now, t_l, t_pos, t_bone = train(train_loader, model, optimizer,
                                           lr_now=lr_now, input_n=input_n, output_n=output_n, device=device,
                                           max_norm=opt.max_norm,
                                           is_cuda=is_cuda,
                                           all_n=all_n)
        print("train_loss:", t_l)
        ret_log = np.append(ret_log, [lr_now, t_l, t_pos, t_bone])
        head = np.append(head, ['lr', 't_l', 't_pos', 't-bone'])
        v_p, v_bone = val(val_loader, model, input_n=input_n, output_n=output_n, device=device, is_cuda=is_cuda,
                          all_n=all_n)
        print("val_loss:", v_p)
        ret_log = np.append(ret_log, [v_p, v_bone])
        head = np.append(head, ['v_p', 'v_bone'])

        test_position = test(train_loader=test_dip_loader, model=model, device=device, is_cuda=is_cuda,
                             input_n=input_n, output_n=output_n
                             )
        test_err = test_position.sum()
        position = 'po_'
        # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
        ret_log = np.append(ret_log, test_position)
        head = np.append(head,
                         [position + '5', position + '17', position + '23',
                          position + '35', position + '36',
                          position + '41', position + '47',
                          position + '53', position + '59'
                          ])
        for act in actions:
            test_position = test(train_loader=test_loader[act], model=model, device=device, is_cuda=is_cuda,
                                 input_n=input_n, output_n=output_n
                                 )
            position = act + 'po_'

            # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
            ret_log = np.append(ret_log, test_position)
            head = np.append(head,
                             [position + '5', position + '17', position + '23',
                              position + '35', position + '36',
                              position + '41', position + '47',
                              position + '53', position + '59'
                              ])
        if not np.isnan(v_p):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_p < err_best  # err_best=10000
            err_best = min(v_p, err_best)
        else:
            is_best = False
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if epoch == start_epoch:
            df.to_csv(ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

        file_name = ['ckpt_' + str(script_name) + str(epoch) + '_best.pth.tar', 'ckpt_']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         # 'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt,
                        file_name=file_name)


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


def train(train_loader, model, optimizer, lr_now, input_n, output_n, device, max_norm, is_cuda, all_n):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    t_pos = utils.AccumLoss()
    t_bone = utils.AccumLoss()
    # 固定句式 在训练模型时会在前面加上
    model.train()
    # input_acc[item], self.input_ori[item], self.out_poses[item], self.out_joints[item]
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            batch_size = input_acc.shape[0]  # 16
            pad_idx = np.repeat([input_n - 1], output_n)
            i_idx = np.append(np.arange(0, input_n), pad_idx)
            input_ori = input_ori
            input_acc = input_acc[:, i_idx, :]
            input_ori = input_ori[:, i_idx, :]
            # out_joints:(32,24,3,frames)
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            model_input = get_dct(input_ori_acc, input_n)
            if is_cuda:
                # print("x")
                model_input = model_input.to(device).float()
                out_joints = out_joints.to(device).float()
            y_out = model(model_input)  # 32,72,50
            # 计算loss
            pred_3d, targ_3d = get_idct(y_out=y_out, out_joints=out_joints, device=device)

            mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))
            loss = mean_3d_err
            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
            # angle_loss = loss_func.angle_loss(y_out, out_poses)
            t_l.update(loss.cpu().data.numpy() * batch_size, batch_size)
            t_pos.update(loss.cpu().data.numpy() * batch_size, batch_size)
            t_bone.update(loss.cpu().data.numpy() * batch_size, batch_size)
    return lr_now, t_l.avg, t_pos.avg, t_bone.avg


def val(train_loader, model, input_n, output_n, device, is_cuda, all_n):
    print("进入val")
    t_3d = utils.AccumLoss()
    t_angle = utils.AccumLoss()

    model.eval()
    with torch.no_grad():
        for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
            if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                    and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
                batch_size = input_acc.shape[0]  # 16
                pad_idx = np.repeat([input_n - 1], output_n)
                i_idx = np.append(np.arange(0, input_n), pad_idx)
                input_ori = input_ori
                input_acc = input_acc[:, i_idx, :]
                input_ori = input_ori[:, i_idx, :]
                # out_joints:(32,24,3,frames)
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                model_input = get_dct(input_ori_acc, input_n)
                if is_cuda:
                    # print("x")
                    model_input = model_input.to(device).float()
                    out_joints = out_joints.to(device).float()
                y_out = model(model_input)  # 32,72,50
                # 计算loss
                pred_3d, targ_3d = get_idct(y_out=y_out, out_joints=out_joints, device=device)
                mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

                loss = mean_3d_err
                t_3d.update(loss.cpu().data.numpy() * batch_size, batch_size)
                t_angle.update(loss.cpu().data.numpy() * batch_size, batch_size)
    return t_3d.avg, t_angle.avg


def test(train_loader, model, input_n, output_n, device, is_cuda):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    eval_frame = [36, 41, 47, 53, 59]
    t_posi = np.zeros(len(eval_frame))  # 6位
    model.eval()
    with torch.no_grad():
        for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
            if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                    and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
                batch_size = input_acc.shape[0]  # 16
                pad_idx = np.repeat([input_n - 1], output_n)
                i_idx = np.append(np.arange(0, input_n), pad_idx)
                input_ori = input_ori
                input_acc = input_acc[:, i_idx, :]
                input_ori = input_ori[:, i_idx, :]
                # out_joints:(32,24,3,frames)
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                model_input = get_dct(input_ori_acc, input_n)
                # print(model_input.shape)
                if is_cuda:
                    # print("x")
                    model_input = model_input.to(device).float()
                    out_joints = out_joints.to(device).float()
                y_out = model(model_input)  # 32,72,50
                batch, frame, node, dim = out_joints.data.shape
                # print(out_joints.shape, y_out.shape)
                _, idct_m = get_dct_matrix(frame)
                idct_m = torch.from_numpy(idct_m).to(torch.float32).to(device)
                outputs_t = y_out.view(-1, frame).permute(1, 0)
                # 50,32*24*3
                outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
                outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
                # 32,72,50

                for k in np.arange(0, len(eval_frame)):  # 6
                    j = eval_frame[k]

                    test_out, test_joints = outputs_p3d[:, j, :, :], out_joints[:, j, :, :]
                    t_posi[k] += loss_func.position_loss(test_out, test_joints).cpu().data.numpy() * batch * 100
                N += batch

    return t_posi / N


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


dip_eval = ['01', '02', '03', '04', '05']


def eval_dip(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n

    device = torch.device('cpu')
    is_cuda = True
    print(">>>is_cuda ", device)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.GCN(input_feature=all_n, hidden_feature=128, p_dropout=0.5,
                        num_stage=12, node_n=72)
    model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # data loading
    print(">>> loading data")
    model_path_len = "F:/实验对比/new/Joint/GCN-DCT" \
                     "/ckpt_main_gcntestnnnnn_n36_out24_dctn607_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_path_len))

    ckpt = torch.load(model_path_len, map_location='cpu')

    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    # data loading
    print(">>> loading data")
    data_path = 'C:/Gtrans/dataset/dataset/dip/dip_test/04'
    test_dataset = amass(path_to_data=data_path, input_n=input_n, output_n=output_n,
                         split=0)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=512,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    print(">>> test data {}".format(test_loader.__len__()))
    test_position = test(train_loader=test_loader, model=model, device=device, is_cuda=is_cuda,
                         input_n=input_n, output_n=output_n)
    print(test_position)


if __name__ == "__main__":
    option = Options().parse()
    # main(option)
    eval_dip(option)
    # main_total_pose(option)
# 查看网络的每一个参数
# print("=============更新之后===========")
# for name, parms in model.named_parameters():
#     print('-->name:', name)
#     print('-->para:', parms)
#     print('-->grad_requirs:', parms.requires_grad)
#     print('-->grad_value:', parms.grad)
#     print("===")
# print(optimizer)
# print("max_norm", max_norm)
