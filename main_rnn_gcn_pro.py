# coding=utf-8
import torch
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from utils.load_total import amass_rnn as total
from utils.amass_6_node import amass_rnn as amass
import model.RNN_best as nnmodel
from utils.opt import Options
from utils import loss_func, utils_utils as utils

actions = ['acting', 'rom', 'walking', 'freestyle']
torch.backends.cudnn.enabled = False


def main(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n
    ignore = opt.ignore
    loss_weight = opt.loss_weight
    adjs = opt.adjs
    start_epoch = 00
    err_best = 10000
    lr_now = opt.lr * 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:去掉两个GCN块
    # new_3:只看后半段预测部分的
    script_name += "AA111_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.Predict_imu(input_frame_len=input_n, output_frame_len=all_n, input_size=72,
                                mid_size=18, output_size=72, adjs=adjs, device=device,
                                dropout=0.2)
    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    # data_amass_dir
    # data_dip_dir
    # data_xt_dip_dir
    # data_dip_total_short_dir
    # data_xt_dip_total_dir
    train_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                          split=0)
    val_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                        split=1)
    test_dip_dataset = amass(path_to_data=opt.data_xt_dip_total_dir, input_n=input_n, output_n=output_n,
                             split=2)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=5,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=512,  # 128
        shuffle=False,
        num_workers=3,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)

    test_dip_loader = DataLoader(
        dataset=test_dip_dataset,
        batch_size=512,  # 128
        shuffle=False,
        num_workers=3,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)
    test_loader = dict()
    for act in actions:
        test_dataset = total(path_to_data=opt.data_xt_total_dir, act=act, input_n=input_n, output_n=output_n,
                             split=2)
        test_loader[act] = DataLoader(
            dataset=test_dataset,
            batch_size=1024,  # 128
            shuffle=False,
            num_workers=1,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
            pin_memory=False)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))  # 32178
    print(">>> validation data {}".format(val_dataset.__len__()))  # 1271
    print(">>> test data {}".format(test_dip_dataset.__len__()))
    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print('=====================================')
        print('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        Ir_now, t_l, t_pos, t_pos_0, t_bone = train(train_loader, model, optimizer, loss_weight=loss_weight,
                                                    device=device,
                                                    input_n=input_n,
                                                    output_n=output_n,
                                                    lr_now=lr_now, max_norm=opt.max_norm, all_n=all_n)
        print("epoch结束")
        print("train_loss:", t_l, "position", t_pos, "posi_0", t_pos_0, "bone", t_bone)
        ret_log = np.append(ret_log, [lr_now, t_l, t_pos, t_pos_0, t_bone])
        head = np.append(head, ['lr', 't_l', 't_pos_weight', 't_pos_0', 't-bone'])
        # validation
        v_p, v_p_0, v_bone = val(val_loader, model, device=device, loss_weight=loss_weight,
                                 input_n=input_n,
                                 output_n=output_n,
                                 all_n=all_n)
        print("val_loss:", v_p, "posi_0", v_p_0)
        ret_log = np.append(ret_log, [v_p, v_p_0, v_bone])
        head = np.append(head, ['v_p', 'v_p_0', 'v_bone'])
        # test
        test_position = test(train_loader=test_dip_loader, device=device, model=model,
                             ignore=ignore,
                             input_n=input_n, all_n=all_n)
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
            test_position = test(train_loader=test_loader[act], device=device, model=model,
                                 ignore=ignore,
                                 input_n=input_n, all_n=all_n)
            position = act + 'po_'

            # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
            ret_log = np.append(ret_log, test_position)
            head = np.append(head,
                             [position + '5', position + '17', position + '23',
                              position + '35', position + '36',
                              position + '41', position + '47',
                              position + '53', position + '59'
                              ])
        if not np.isnan(test_err):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = test_err < err_best  # err_best=10000
            err_best = min(test_err, err_best)
        else:
            is_best = False
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if epoch == start_epoch:

            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if is_best:
            file_name = ['ckpt_' + str(script_name) + '_best.pth.tar', 'ckpt_']
            utils.save_ckpt({'epoch': epoch + 1,
                             'lr': lr_now,
                             # 'err': test_e[0],
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            ckpt_path=opt.ckpt,
                            file_name=file_name)


def train(train_loader, model, optimizer, device, loss_weight, input_n, output_n, lr_now, max_norm, all_n):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    t_pos = utils.AccumLoss()
    t_pos_0 = utils.AccumLoss()
    t_bone = utils.AccumLoss()
    # 固定句式 在训练模型时会在前面加上
    model.train()
    # input_acc[item], self.input_ori[item], self.out_poses[item], self.out_joints[item]
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            n = input_ori.shape[0]  # 16
            if torch.cuda.is_available():
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()

            y_out = model(input_ori, input_acc)
            posi_loss = loss_func.position_loss(y_out, out_joints)
            bone_loss = loss_func.bone_loss(y_out, out_joints)
            loss = posi_loss + bone_loss * 0.2
            # print(loss)

            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()

            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值

            t_l.update(loss.cpu().data.numpy() * n, n)
            t_pos.update(posi_loss.cpu().data.numpy() * n, n)
            t_pos_0.update(posi_loss.cpu().data.numpy() * n, n)
            t_bone.update(bone_loss.cpu().data.numpy() * n, n)
    return lr_now, t_l.avg, t_pos.avg, t_pos_0.avg, t_bone.avg


def val(train_loader, model, device, input_n, output_n, loss_weight, all_n):
    print("进入val")
    t_posi = utils.AccumLoss()
    t_posi_0 = utils.AccumLoss()
    t_bone = utils.AccumLoss()
    model.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:

            n = input_ori.shape[0]  # 16

            if torch.cuda.is_available():
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()
                loss_weight = loss_weight.to(device).float()
            y_out = model(input_ori, input_acc)
            posi_loss = loss_func.position_loss_gai(y_out, out_joints, loss_weight=loss_weight)
            posi_loss_0 = loss_func.position_loss(y_out, out_joints)
            bone_loss = loss_func.bone_loss(y_out, out_joints)
            t_posi.update(posi_loss.cpu().data.numpy() * n, n)
            t_posi_0.update(posi_loss_0.cpu().data.numpy() * n, n)
            t_bone.update(bone_loss.cpu().data.numpy() * n, n)
    return t_posi.avg, t_posi_0.avg, t_bone.avg


def test(train_loader, model, device, ignore, input_n, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    # eval_frame = [2, 5, 8, 11, 14, 29]
    # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
    eval_frame = [5, 17, 23, 35, 36, 41, 47, 53, 59]

    t_posi = np.zeros(len(eval_frame))  # 6位
    model.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            if torch.cuda.is_available():
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()
            # model要改
            y_out = model(input_ori, input_acc)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_joints = y_out[:, j, :], out_joints[:, j, :]
                t_posi[k] += torch.mean(torch.norm(test_out - test_joints, 2, 2)).cpu().data.numpy() * batch * 100

            N += batch

    return t_posi / N


if __name__ == "__main__":
    option = Options().parse()
    main(option)
    # eval(option)
    # demo_benji(option)
