#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-identifier]
"""

import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import torch
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir

def freeze_bn(net):
    count = 0
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval() 
            module.weight.requires_grad = False
            module.bias.requires_grad = False


def load_my_state_dict(old_state, state_dict):

    own_state = old_state.state_dict()
    for name, param in state_dict.items():
        
        if name == 'backbone.backbone.conv1.weight':#not in own_state:
            own_state[name].copy_(torch.cat((param,torch.mean(param,dim=1,keepdim=True)),dim=1))
        else:
            own_state[name].copy_(param)

def main():
    args = docopt(__doc__)
    config_file = "config/wireframe.yaml" #args["<yaml-config>"] or 
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from
    resume_from = '/home/pebert/190418-201834-f8934c6-lr4d10-312k.pth.tar'

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = '5' #args["--devices"] #'0'
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset

    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(datadir, split="train")[0]
    # sys.exit(0)

    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="train"),
        shuffle=True,
        batch_size=M.batch_size,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )
    epoch_size = len(train_loader)
    # print("epoch_size (train):", epoch_size)
    # print("epoch_size (valid):", len(val_loader))

    if resume_from:
        checkpoint = torch.load(osp.join(resume_from))#, "checkpoint_latest.pth"))

    # 2. model
    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)
    model = LineVectorizer(model)

    
    if resume_from:
        load_my_state_dict(model,checkpoint["model_state_dict"])
        #model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    freeze_bn(model)

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    #if resume_from:
    #    optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = get_outdir(args["--identifier"]) # 'baseline' resume_from or get_outdir('test') args["--identifier"]
    print("outdir:", outdir)

    try:
        trainer = lcnn.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
