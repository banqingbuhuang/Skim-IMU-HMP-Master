# coding=utf-8
import os
import torch
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas

from utils.amass_6_node import amass_rnn as amass
import model.RNN_6_pose_DIP as nnmodel
from utils.opt import Options
from utils import loss_func, utils_utils as utils

CUDA_LAUNCH_BLOCKING = 1


def main(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n
    ignore = opt.ignore
    loss_weight = opt.loss_weight
    adjs = opt.adjs
    start_epoch = 00
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:去掉两个GCN块
    # new_3:只看后半段预测部分的
    script_name += "1_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                mid_size=54, output_size=216, adjs=adjs,
                                device=device, dropout=0.3)
    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    # data_amass_dir
    # data_xt_dip_dir
    # data_dip_dir
    train_dataset = amass(path_to_data=opt.data_dip_total_dir, input_n=input_n, output_n=output_n,
                          split=0)
    val_dataset = amass(path_to_data=opt.data_dip_total_dir, input_n=input_n, output_n=output_n,
                        split=1)
    test_dip_dataset = amass(path_to_data=opt.data_dip_dir, input_n=input_n, output_n=output_n,
                             split=3)
    test_total_dataset=amass(path_to_data=opt.data_dip_dir, input_n=input_n, output_n=output_n,
                             split=3)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=5,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=1,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)

    test_loader = DataLoader(
        dataset=test_dip_dataset,
        batch_size=256,  # 128
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
        Ir_now, t_l = train(train_loader, model, optimizer, loss_weight,
                            input_n=input_n,
                            output_n=output_n,
                            cuda=device,
                            lr_now=lr_now, max_norm=opt.max_norm, all_n=all_n)
        print("epoch结束")
        print("train_loss:", t_l)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])
        # validation
        v_p = val(val_loader, model, cuda=device, loss_weight=loss_weight,
                  input_n=input_n,
                  output_n=output_n,
                  all_n=all_n)
        print("val_loss:", v_p)
        ret_log = np.append(ret_log, v_p)
        head = np.append(head, 'v_p')
        # test
        test_all, test_full = test(train_loader=test_loader, model=model, cuda=device,
                                   ignore=ignore,
                                   input_n=input_n, all_n=all_n)
        sm, am, pm, me, jitm = 'sip_m', 'angle_m', 'pos_m', 'Mesh_m', 'jitter_m'
        sd, ad, pd, md, jitd = 'sip_d', 'angle_d', 'pos_d', 'Mesh_d', 'jitter_d'
        ret_log = np.append(ret_log, test_full)
        head = np.append(head,
                         [sm + '5', sd, am + '5', ad, pm + '5', pd, me + '5', md, jitm + '5', jitd,
                          sm + '17', sd, am + '17', ad, pm + '17', pd, me + '17', md, jitm + '17', jitd,
                          sm + '23', sd, am + '23', ad, pm + '23', pd, me + '23', md, jitm + '23', jitd,
                          sm + '35', sd, am + '35', ad, pm + '35', pd, me + '35', md, jitm + '35', jitd,
                          sm + '36', sd, am + '36', ad, pm + '36', pd, me + '36', md, jitm + '36', jitd,
                          ])
        if not np.isnan(v_p):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_p < err_best  # err_best=10000
            err_best = min(v_p, err_best)
        else:
            is_best = False
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pandas.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
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


def train(train_loader, model, optimizer, loss_weight, input_n, output_n, cuda, lr_now, max_norm, all_n):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    # 固定句式 在训练模型时会在前面加上
    model.train()
    # input_acc[item], self.input_ori[item], self.out_poses[item], self.out_joints[item]
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            n = input_ori.shape[0]  # 16
            if torch.cuda.is_available():
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_poses = out_poses.to(cuda).float()
                loss_weight = loss_weight.to(cuda).float()
            y_out = model(input_ori, input_acc)
            loss = loss_func.poses_loss_yi(y_out, out_poses)
            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
            t_l.update(loss.cpu().data.numpy() * n, n)
    return lr_now, t_l.avg


def val(train_loader, model, cuda, input_n, output_n, loss_weight, all_n):
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
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_poses = out_poses.to(cuda).float()
                loss_weight = loss_weight.to(cuda).float()
            y_out = model(input_ori, input_acc)
            loss = loss_func.poses_loss_yi(y_out, out_poses)
            t_posi.update(loss.cpu().data.numpy() * n, n)
            t_posi_0.update(loss.cpu().data.numpy() * n, n)
            t_bone.update(loss.cpu().data.numpy() * n, n)
    return t_posi.avg


def test(train_loader, model, cuda, ignore, input_n, all_n):
    print("进入test")
    N = 0
    # 100,200,300,400,500,1000
    eval_frame = [5, 17, 23, 35, 36]
    test_record = torch.zeros([len(eval_frame), 5, 2])
    test_all = torch.zeros([len(eval_frame), 4])

    model.eval()
    evaluator = loss_func.PoseEvaluator(official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl")
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            if torch.cuda.is_available():
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_poses = out_poses.to(cuda).float()
            # model要改
            y_out = model(input_ori, input_acc)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_poses = y_out[:, j, :], out_poses[:, j, :]
                test_out = test_out.reshape(-1, 9).contiguous()
                test_poses = test_poses.reshape(-1, 9).contiguous()
                angle = loss_func.angle_error(pose_p=test_out, pose_t=test_poses)
                print("angle", angle)
                test_record[k] += (evaluator.eval(test_out, test_poses)) * batch
                test_all[k] += (evaluator.eval_all(test_out, test_poses)) * batch
            N += batch
    return (test_all / N).flatten(0), (test_record / N).flatten(0)


def eval(opt):
    input_n = 36
    output_n = 24
    all_n = input_n + output_n
    ignore = opt.ignore
    loss_weight = opt.loss_weight
    adjs = opt.adjs
    start_epoch = 00
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:去掉两个GCN块
    # new_3:只看后半段预测部分的
    script_name += "1_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                mid_size=54, output_size=216, adjs=adjs,
                                device=device, dropout=0.3)
    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # new_3:将up变成LSTM

    model_pro_path = "/home/xt/Skim-IMU-master/checkpoint/test/ckpt_main_6_pose0_in36_out24_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_pro_path))
    if is_cuda:
        ckpt = torch.load(model_pro_path)
    else:
        ckpt = torch.load(model_pro_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    test_dataset = amass(path_to_data=opt.data_xt_dip_dir, input_n=input_n, output_n=output_n,
                         split=3)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    model.eval()
    N = 0
    eval_frame = [5, 17, 23, 35, 36]
    test_record = torch.zeros([len(eval_frame), 5, 2])
    test_all = torch.zeros([len(eval_frame), 4])
    evaluator = loss_func.PoseEvaluator(official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl"
                                        , smpl_folder="/data/xt/body_models/vPOSE/models")
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            if torch.cuda.is_available():
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_poses = out_poses.to(device).float()
            # model要改
            y_out = model(input_ori, input_acc)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_poses = y_out[:, j, :], out_poses[:, j, :]
                test_out = test_out.reshape(-1, 9).contiguous()
                test_poses = test_poses.reshape(-1, 9).contiguous()
                angle = loss_func.angle_error(pose_p=test_out, pose_t=test_poses)
                print("angle", angle)
                # test_record[k] += (evaluator.eval(test_out, test_poses)) * batch
                print((evaluator.eval(test_out, test_poses)))
                # test_all[k] += (evaluator.eval_all(test_out.cpu(), test_poses.cpu())) * batch
                print((evaluator.eval_all(test_out.cpu(), test_poses.cpu())))
            N += batch


if __name__ == "__main__":
    option = Options().parse()
    # main(option)
    eval(option)
    # demo_benji(option)
