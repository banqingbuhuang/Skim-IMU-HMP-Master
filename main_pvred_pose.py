# coding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from utils.load_total import amass_rnn as total
from utils.amass_seq import amass_rnn as amass
import model.pvred_enc as nnmodel
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
    ckpt = opt.ckpt + '_residual'
    ignore = opt.ignore
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:测试total_capture
    script_name += "pose_new_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    # input_seq,
    model = nnmodel.Encoder_Decoder(input_size=216, hidden_size=1024, num_layer=1, rnn_unit='gru',
                                    residual=True, out_dropout=0.3, std_mask=True, veloc=True, pos_embed=True
                                    , pos_embed_dim=96, device=device)
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
        batch_size=512,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=5,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=2048,  # 128
        shuffle=False,
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)

    test_dip_loader = DataLoader(
        dataset=test_dip_dataset,
        batch_size=2048,  # 128
        shuffle=False,
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
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
        print(">>> test data {}".format(test_loader[act].__len__()))
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

        test_position = test(test_dip_loader, model, input_n=input_n, output_n=output_n, device=device, is_cuda=is_cuda,
                             all_n=all_n
                             )
        position = 'po_'
        # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
        ret_log = np.append(ret_log, test_position)
        sm, am, pm, me = 'sip_m', 'angle_m', 'pos_m', 'Mesh_m'
        head = np.append(head,
                         [
                             # am + '17', me + '17',
                             am + '36', me + '36',
                             am + '41', me + '41',
                             am + '47', me + '47',
                             am + '53', me + '53',
                             am + '59', me + '59',
                         ])
        for act in actions:
            test_position = test(train_loader=test_loader[act], model=model, input_n=input_n,
                                 output_n=output_n, device=device, is_cuda=is_cuda,
                                 all_n=all_n
                                 )
            position = act + 'po_'
            am = act + am
            me = act + me
            # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
            ret_log = np.append(ret_log, test_position)
            head = np.append(head,
                             [
                                 # am + '17', me + '17',
                                 am + '36', me + '36',
                                 am + '41', me + '41',
                                 am + '47', me + '47',
                                 am + '53', me + '53',
                                 am + '59', me + '59',
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
        file_name = ['ckpt_' + str(script_name) + str(epoch) + '.pth.tar', 'ckpt_']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         # 'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now, input_n, output_n, device, max_norm, is_cuda, all_n):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    t_pos = utils.AccumLoss()
    t_bone = utils.AccumLoss()
    loss_f = nn.SmoothL1Loss()
    # 固定句式 在训练模型时会在前面加上
    model.train()
    # input_acc[item], self.input_ori[item], self.out_poses[item], self.out_joints[item]
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            batch_size = input_acc.shape[0]  # 16
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            encoder_input = input_ori_acc[:, 0:input_n - 1, :].transpose(0, 1)
            target_input = input_ori_acc[:, input_n - 1:].transpose(0, 1)
            out_poses = out_poses[:, input_n:].reshape(batch_size, output_n, -1)
            if is_cuda:
                encoder_input = encoder_input.to(device).float()
                target_input = target_input.to(device).float()
                out_poses = out_poses.to(device).float()
            outputs_enc, outputs_dec = model(encoder_input, target_input)  # 32,72,50
            outputs_dec = outputs_dec.transpose(0, 1)
            loss = loss_f(outputs_dec, out_poses)
            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
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
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                encoder_input = input_ori_acc[:, 0:input_n - 1, :].transpose(0, 1)
                target_input = input_ori_acc[:, input_n - 1:].transpose(0, 1)
                out_poses = out_poses[:, input_n:].reshape(batch_size, output_n, -1)
                if is_cuda:
                    encoder_input = encoder_input.to(device).float()
                    target_input = target_input.to(device).float()
                    out_poses = out_poses.to(device).float()
                outputs_enc, outputs_dec = model(encoder_input, target_input)  # 32,72,50
                outputs_dec = outputs_dec.transpose(0, 1)
                loss = loss_func.poses_loss(outputs_dec, out_poses)
                t_3d.update(loss.cpu().data.numpy() * batch_size, batch_size)
                t_angle.update(loss.cpu().data.numpy() * batch_size, batch_size)
    return t_3d.avg, t_angle.avg


def test(train_loader, model, input_n, output_n, device, is_cuda, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400
    eval_frame = [0, 5, 11, 17, 23]
    test_all = torch.zeros([len(eval_frame), 2])
    # official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl",
    #                                         smpl_folder="/data/xt/body_models/vPOSE/models"
    model.eval()
    evaluator = loss_func.PoseEvaluator(official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl",
                                        smpl_folder="/data/xt/body_models/vPOSE/models")
    with torch.no_grad():
        for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
            if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                    and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
                batch_size = input_acc.shape[0]  # 16
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                encoder_input = input_ori_acc[:, 0:input_n - 1, :].transpose(0, 1)
                target_input = input_ori_acc[:, input_n - 1:].transpose(0, 1)
                out_poses = out_poses[:, input_n:]
                if is_cuda:
                    encoder_input = encoder_input.to(device).float()
                    target_input = target_input.to(device).float()
                    out_poses = out_poses.to(device).float()
                outputs_enc, outputs_dec = model(encoder_input, target_input)  # 32,72,50
                outputs_dec = outputs_dec.transpose(0, 1)
                batch, frame, node, _, _ = out_poses.data.shape
                y_out = outputs_dec.view(batch, frame, node, -1)
                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    test_out, test_poses = y_out[:, j, :], out_poses[:, j, :]
                    test_all[k] += (evaluator.eval_all(test_out.cpu(), test_poses.cpu())) * batch
                N += batch
    return (test_all / N).flatten(0)


def main_total_pose(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    ignore = opt.ignore
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:测试total_capture
    script_name += "_total_pose_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    # input_seq,
    model = nnmodel.Seq2SeqModel(input_seq=input_n, target_seq=output_n,
                                 rnn_size=1024, input_size=216, device=device)

    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    train_dataset = amass(path_to_data=opt.data_xt_total_dir, input_n=input_n, output_n=output_n,
                          split=0)
    val_dataset = amass(path_to_data=opt.data_xt_total_dir, input_n=input_n, output_n=output_n,
                        split=1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=1,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    test_loader = dict()
    for act in actions:
        test_dataset = total(path_to_data=opt.data_xt_total_dir, act=act, input_n=input_n, output_n=output_n,
                             split=3)

        test_loader[act] = DataLoader(
            dataset=test_dataset,
            batch_size=128,  # 128
            shuffle=False,
            num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
            pin_memory=False)
        print(">>> test data {}".format(test_loader[act].__len__()))

    print(">>> train data {}".format(train_dataset.__len__()))  # 32178
    print(">>> validation data {}".format(val_dataset.__len__()))  # 1271
    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print('=====================================')
        print('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        Ir_now, t_l, t_pos, t_bone = train_pose(train_loader, model, optimizer,
                                                lr_now=lr_now, input_n=input_n, output_n=output_n, device=device,
                                                max_norm=opt.max_norm,
                                                is_cuda=is_cuda,
                                                all_n=all_n)
        print("train_loss:", t_l)
        ret_log = np.append(ret_log, [lr_now, t_l, t_pos, t_bone])
        head = np.append(head, ['lr', 't_l', 't_pos', 't-bone'])
        v_p, v_bone = val_pose(val_loader, model, input_n=input_n, output_n=output_n, device=device, is_cuda=is_cuda,
                               all_n=all_n)
        print("val_loss:", v_p)
        ret_log = np.append(ret_log, [v_p, v_bone])
        head = np.append(head, ['v_p', 'v_bone'])
        for act in actions:
            test_full = test_pose(test_loader[act], model, input_n=input_n, output_n=output_n, ignore=ignore,
                                  device=device,
                                  is_cuda=is_cuda,
                                  all_n=all_n)
            sm, am, pm, me = 'sip_m', 'angle_m', 'pos_m', 'Mesh_m'
            ret_log = np.append(ret_log, test_full)
            head = np.append(head,
                             [sm + '5', am + '5', pm + '5', me + '5',
                              sm + '17', am + '17', pm + '17', me + '17',
                              sm + '35', am + '35', pm + '35', me + '35',
                              sm + '41', am + '41', pm + '41', me + '41',
                              sm + '47', am + '47', pm + '47', me + '47',
                              sm + '53', am + '53', pm + '53', me + '53',
                              sm + '59', am + '59', pm + '59', me + '59',
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


def train_pose(train_loader, model, optimizer, lr_now, input_n, output_n, device, max_norm, is_cuda, all_n):
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
            batch, frame, node, _ = out_joints.shape
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            input_ori_acc = torch.cat([input_ori_acc, input_ori_acc, input_ori_acc], dim=2)
            pad_idx = np.repeat([input_n - 1], output_n)
            i_idx = np.append(np.arange(0, input_n), pad_idx)
            encoder_input = input_ori_acc[:, 0:input_n - 1, :]
            decoder_input = input_ori_acc[:, i_idx, :]

            if is_cuda:
                encoder_input = encoder_input.to(device).float()
                decoder_input = decoder_input.to(device).float()
                out_poses = out_poses.to(device).float()

            y_out = model(encoder_input, decoder_input).contiguous()  # 32,72,50
            loss = loss_func.poses_loss(y_out, out_poses)
            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
            t_l.update(loss.cpu().data.numpy() * batch_size, batch_size)
            t_pos.update(loss.cpu().data.numpy() * batch_size, batch_size)
            t_bone.update(loss.cpu().data.numpy() * batch_size, batch_size)
    return lr_now, t_l.avg, t_pos.avg, t_bone.avg


def val_pose(train_loader, model, input_n, output_n, device, is_cuda, all_n):
    print("进入val")
    t_3d = utils.AccumLoss()
    t_angle = utils.AccumLoss()

    model.eval()
    with torch.no_grad():
        for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
            if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                    and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
                batch_size = input_acc.shape[0]  # 16
                batch, frame, node, _ = out_joints.shape
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                input_ori_acc = torch.cat([input_ori_acc, input_ori_acc, input_ori_acc], dim=2)

                pad_idx = np.repeat([input_n - 1], output_n)
                i_idx = np.append(np.arange(0, input_n), pad_idx)
                encoder_input = input_ori_acc[:, 0:input_n - 1, :]
                decoder_input = input_ori_acc[:, i_idx, :]
                if is_cuda:
                    encoder_input = encoder_input.to(device).float()
                    decoder_input = decoder_input.to(device).float()
                    out_poses = out_poses.to(device).float()
                y_out = model(encoder_input, decoder_input).contiguous()  # 32,72,50
                loss = loss_func.poses_loss(y_out=y_out, out_poses=out_poses)
                t_3d.update(loss.cpu().data.numpy() * batch_size, batch_size)
                t_angle.update(loss.cpu().data.numpy() * batch_size, batch_size)
    return t_3d.avg, t_angle.avg


def test_pose(train_loader, model, input_n, output_n, ignore, device, is_cuda, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    eval_frame = [5, 17, 35, 41, 47, 53, 59]
    # eval_frame = [3, 2, 5, 8, 11, 14, 17, 20, 23]
    test_record = torch.zeros([len(eval_frame), 4])

    model.eval()
    evaluator = loss_func.PoseEvaluator(official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl"
                                        , smpl_folder="/data/xt/body_models/vPOSE/models")
    with torch.no_grad():
        for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
            if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                    and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
                input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
                input_ori_acc = torch.cat([input_ori_acc, input_ori_acc, input_ori_acc], dim=2)
                batch, frame, node, _ = out_joints.shape
                pad_idx = np.repeat([input_n - 1], output_n)
                i_idx = np.append(np.arange(0, input_n), pad_idx)
                encoder_input = input_ori_acc[:, 0:input_n - 1, :]
                decoder_input = input_ori_acc[:, i_idx, :]
                if is_cuda:
                    encoder_input = encoder_input.to(device).float()
                    decoder_input = decoder_input.to(device).float()
                    out_poses = out_poses.to(device).float()
                y_out = model(encoder_input, decoder_input).contiguous()  # 32,72,50
                batch = y_out.data.shape[0]
                for k in np.arange(0, len(eval_frame)):  # 6
                    j = eval_frame[k]

                    test_out, test_poses = y_out[:, j, ], out_poses[:, j, ]
                    test_record[k] += (evaluator.eval_all(test_out.cpu(), test_poses.cpu())) * batch
                N += batch
    print((test_record / N).flatten(0), "退出test")
    return (test_record / N).flatten(0)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
    # main_total(option)
    # demo(option)
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
