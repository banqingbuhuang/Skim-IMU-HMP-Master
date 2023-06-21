import torch
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import smplx
from copy import copy

def main():

def load_data(grab_path_sample):
    amass_npz_fname = "C:/Gtrans/dataset/GRAB_unzip/grab/s1/airplane_lift.npz"
    seq_data = parse_npz(amass_npz_fname)
    frame_names = []
    body_data = {
        'global_orient': [], 'body_pose': [], 'transl': [],
        'right_hand_pose': [], 'left_hand_pose': [],
        'jaw_pose': [], 'leye_pose': [], 'reye_pose': [],
        'expression': [], 'fullpose': [],
        'contact': [], 'verts': []
    }
    obj_name = seq_data.obj_name
    sbj_id = seq_data.sbj_id
    n_comps = seq_data.n_comps
    gender = seq_data.gender

    frame_mask = filter_contact_frames(seq_data)

    # total selectd frames
    T = frame_mask.sum()

    sbj_params = prepare_params(seq_data.body.params, frame_mask)

    append2dict(body_data, sbj_params)
    sbj_vtemp = load_sbj_verts(sbj_id, seq_data)
def load_sbj_verts(grab_path,sbj_info, sbj_id, seq_data):

    mesh_path = os.path.join(grab_path, '..', seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp
def filter_contact_frames(self, seq_data):
    if self.cfg.only_contact:
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
    else:
        frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)
    return frame_mask
def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)
def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}
def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))