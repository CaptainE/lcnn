import glob
import json
import math
import os
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M

# use for curvature and edge
import cv2


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist2 = glob.glob(f"/home/pebert/taskonomy/dataset/{split}/*_edge.png")
        filelist3 = glob.glob(f"/home/pebert/taskonomy/dataset/{split}/*_curvature.png")
        filelist.sort()
        filelist2.sort()
        filelist3.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist
        self.filelist2 = filelist2
        self.filelist3 = filelist3

    def __len__(self):
        return len(self.filelist)

    def read_edge_curve_img(self, idx, types='curve'):
        if types=='curve':
            location = self.filelist3[idx]
        else:
            location = self.filelist2[idx]

        #'+folder+'/'+num+'_curvature.png'
        img = cv2.imread(location, cv2.IMREAD_UNCHANGED)
                
        scale_percent = 200 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        return np.array(resized, dtype=np.float32)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"
        image = io.imread(iname).astype(float)[:, :, :3]
        if "a1" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image = (image - M.image.mean) / M.image.stddev
        image = np.rollaxis(image, 2).copy()


        # /home/pebert/bts/pytorch/result_wireframe/raw/
        dname = '/home/pebert/bts/pytorch/result_wireframe/raw/'+self.filelist[idx][:-10].split('/')[-1].replace("_a0", "").replace("_a1", "") + "_depth.png"

        pred_depth = io.imread(dname).astype(float)
        # Should not be needed with new clear depth images
        #/1000
        #pred_depth[pred_depth < 1e-3] = 1e-3
        #pred_depth[pred_depth > 10] = 10
        #pred_depth[np.isinf(pred_depth)] = 10
        #pred_depth[np.isnan(pred_depth)] = 1e-3
        #pred_depth = pred_depth/10*255
        edge = self.read_edge_curve_img(idx, types='edge')
        curve = self.read_edge_curve_img(idx, types='curve')

        image = np.concatenate((image,pred_depth.reshape(1,512,512)),axis=0)
        image = np.concatenate((image,edge.reshape(1,512,512)),axis=0)
        image = np.concatenate((image,curve.reshape(1,512,512)),axis=0)



        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = np.random.permutation(npz["lpos"])[: M.n_stc_posl]
            lneg = np.random.permutation(npz["lneg"])[: M.n_stc_negl]
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * M.use_cood,
                ldir * M.use_slop,
                lpre[:, :, 2],
            ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
                "lpre_feat": torch.from_numpy(feat),
            }

        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )
