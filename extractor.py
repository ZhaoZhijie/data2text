# -*- coding: utf-8 -*-
import h5py
from configargparse import ArgumentParser
import torch
import math
import numpy as np
import gc
from torch import nn
import sys

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1000, backupCount=10)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

init_logger(log_file="extractor.log")


class ParallelTable(nn.ModuleList):
    def __init__(self):
        super(ParallelTable, self).__init__()
    
    def add(self, mod):
        self.append(mod)

    def forward(self, source):
        outputs = []
        for i in range(len(source)):
            outputs.append(self[i](source[i]))
        return outputs

class JoinTable(nn.Module):
    def __init__(self, dim):
        super(JoinTable, self).__init__()
        self.dim = dim

    def forward(self, vecs):
        output = torch.cat(vecs, self.dim)
        return torch.cat(vecs, self.dim)

class ConcatTable(nn.ModuleList):
    def __init__(self):
        super(ConcatTable, self).__init__()

    def add(self, mod):
        self.append(mod)

    def forward(self, source):
        outputs = []
        for i in range(len(self)):
            outputs.append(self[i](source))
        return outputs

class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, self.dim)[0]

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return nn.functional.relu(x)

class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x):
        return nn.functional.softmax(x)

class DebugLayer(nn.Module):
    def __init__(self, name="DebugLayer"):
        super(DebugLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print("{} x[0] shape {} dtype{}".format(self.name, x[0].shape, x[0].dtype))
        return x

class Transpose(nn.Module):
    def __init__(self, dims):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        output = torch.transpose(x, self.dims[0], self.dims[1])
        return output

parser = ArgumentParser(description='extractor.py')

parser.add('-datafile', '--datafile', default='roto-ie.h5', help='path to hdf5 file containing train/val data')
parser.add('-batchsize', '--batchsize', default=32, type=int, help='batch size')
parser.add('-embed_size', '--embed_size', default=200, type=int, help='size of embeddings')
parser.add('-num_filters', '--num_filters', default=200, type=int, help='number of convolutional filters')
parser.add('-conv_fc_layer_size', '--conv_fc_layer_size', default=500, type=int, help='size of fully connected layer in convolutional model')
parser.add('-blstm_fc_layer_size', '--blstm_fc_layer_size', default=700, type=int, help='size of fully connected layer in BLSTM model')
parser.add('-dropout', '--dropout', default=0.5, type=float, help='dropout rate')
parser.add('-uniform_init', '--uniform_init', default=0.1, type=float, help='init in params in this range')
parser.add('-lr', '--lr', default=0.7, type=float, help='learning rate')
parser.add('-lr_decay', '--lr_decay', default=0.5, type=float, help='decay factor')
parser.add('-clip', '--clip', default=5, help='clip grads so they do not exceed this')
parser.add('-seed', '--seed', default=3435, type=int, help='Random seed')
parser.add('-epochs', '--epochs', default=10, type=int, help='training epochs')
parser.add('-gpuid', '--gpuid', default=1, type=int, help='gpu idx')
parser.add('-savefile', '--savefile', default='', help='path to save model to')
parser.add('-preddata', '--preddata', default='', help='path to hdf5 file containing candidate relations from generated data')
parser.add('-dict_pfx', '--dict_pfx', default='', help='prefix of .dict and .labels files')
parser.add('-ignore_idx', '--ignore_idx', default=11, type=int, help='idx of NONE class in *.labels file')
parser.add('-just_eval', '--just_eval', default=False, action="store_true", help='just eval generations')
parser.add('-lstm', '--lstm', default=False, action="store_true", help='use a BLSTM rather than a convolutional model')
parser.add('-geom', '--geom', default=False, action="store_true", help='average models geometrically')
parser.add('-test', '--test', default=False, action="store_true", help='use test data')

opt = parser.parse_args()
if torch.cuda.is_available():
  print("gpu is ok")
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


#为了测试方便这里将参数直接赋值
class Args():
  def __init__(self):
    self.datafile = "roto-ie.h5"
    self.batchsize = 32
    self.embed_size = 200
    self.num_filters = 200
    self.conv_fc_layer_size = 500
    self.blstm_fc_layer_size = 700
    self.dropout = 0.5
    self.uniform_init = 0.1
    self.lr = 0.7
    self.lr_decay = 0.5
    self.clip = 5
    self.seed = 3435
    self.epochs = 10
    self.gpuid = 0
    self.savefile = "roto-convie"#roto-blstmie
    self.preddata = ""
    self.dict_pfx = ""
    self.ignore_idx = 11
    self.just_eval = False
    self.lstm = False
    self.geom = False
    self.test = False

def prep_data(batchsize):
    f = h5py.File(opt.datafile, "r")
    trlabels = torch.tensor(np.array(f["trlabels"]))
    perm = torch.randperm(trlabels.shape[0]).to(torch.long)
    trlabels = trlabels[perm]
    trsents = torch.tensor(np.array(f["trsents"]))[perm]
    trlens = torch.tensor(np.array(f["trlens"]))[perm]
    trentdists = torch.tensor(np.array(f["trentdists"]))[perm]
    trnumdists = torch.tensor(np.array(f["trnumdists"]))[perm]
    
    pre = "test" if opt.test else "val"
    valsents = torch.tensor(np.array(f[pre+"sents"]))
    vallens = torch.tensor(np.array(f[pre+"lens"]))
    valentdists = torch.tensor(np.array(f[pre+"entdists"]))
    valnumdists = torch.tensor(np.array(f[pre+"numdists"]))
    vallabels = torch.tensor(np.array(f[pre+"labels"]))
    vallabelnums = vallabels[:, -1]
    vallabels = vallabels[:,:-1]
    f.close()

    psents = None
    plens = None
    pentdists = None
    pnumdists = None
    plabels = None
    pboxrestartidxs = None

    if opt.just_eval and len(opt.preddata) > 0:
        f = h5py.File(opt.preddata, "r")
        psents = torch.tensor(np.array(f["valsents"]))
        plens = torch.tensor(np.array(f["vallens"]))
        pentdists = torch.tensor(np.array(f["valentdists"]))
        pnumdists = torch.tensor(np.array(f["valnumdists"]))
        plabels = torch.tensor(np.array(f["vallabels"]))
        pboxrestartidxs = torch.tensor(np.array(f["boxrestartidxs"]))
        plabelnums = plabels[:, -1]
        plabels = plabels[:,:-1]
        f.close()

    #need to shift negative distances...
    min_entdist = torch.min(trentdists.min(), valentdists.min())
    if isinstance(pentdists, torch.Tensor):
        pentdists.clamp_(min_entdist, trentdists.max())
    min_numdist = torch.min(trnumdists.min(), valnumdists.min())
    if isinstance(pentdists, torch.Tensor):
        pnumdists.clamp_(min_numdist, trnumdists.max())
    trentdists.add_(-min_entdist)
    valentdists.add_(-min_entdist)
    if isinstance(pentdists, torch.Tensor):
        pentdists.add_(-min_entdist)
    trnumdists.add_(-min_numdist)
    valnumdists.add_(-min_numdist)
    if isinstance(pnumdists, torch.Tensor):
        pnumdists.add_(-min_numdist)

    nlabels = trlabels.max().item() + 1

    word_pad = trsents.max()+1
    ent_dist_pad = trentdists.max()+1
    num_dist_pad = trnumdists.max()+1

    def make_batches(sents, lens, entdists, numdists, labels, labelnums=None):
        batches = []
        for i in range(0, sents.size(0), batchsize):
            ub = min(i+batchsize, sents.size(0))
            max_len = lens[i:ub].max()
            for j in range(i, ub):
                if lens[j] < max_len:
                    sents[j, lens[j]:max_len].fill_(word_pad)
                    entdists[j, lens[j]:max_len].fill_(ent_dist_pad)
                    numdists[j, lens[j]:max_len].fill_(num_dist_pad)
            batches.append({
                "sent": sents[i:ub, :max_len],
                "ent_dists": entdists[i:ub, :max_len],
                "num_dists": numdists[i:ub, :max_len],
                "labels": labels[i:ub],
                "labelnums": labelnums[i:ub] if isinstance(labelnums, torch.Tensor) else None
            })
        return batches

    tr_batches = make_batches(trsents, trlens, trentdists, trnumdists, trlabels)
    val_batches = make_batches(valsents, vallens, valentdists, valnumdists, vallabels, vallabelnums)
    pred_batches = None
    if psents:
        pred_batches = make_batches(psents, plens, pentdists, pnumdists, plabels, plabelnums)
    # gc.collect()
    return tr_batches, val_batches, (word_pad, ent_dist_pad, num_dist_pad), nlabels, pred_batches, pboxrestartidxs

def get_dict(finame, invert=False):
    dict_ = {}
    dict_size = 0
    fi = open(finame, "r")
    lines = fi.readlines()
    for line in lines:
        if line != "":
            pieces = line.split()
            if invert:
                dict_[pieces[1]] = pieces[0]
            else:
                dict_[pieces[0]] = int(pieces[1])
            dict_size += 1
    fi.close()
    return dict_, dict_size



def make_conv_model(vocab_sizes, emb_sizes, nlabels, opt):
    par = ParallelTable()
    first_layer_size = 0
    kWs = [2, 3, 5] #kernel widths
    print("emb_sizes",emb_sizes)
    print("vocab_sizes",vocab_sizes)
    for j in range(len(vocab_sizes)):
        if emb_sizes:
            par.add(nn.Embedding(vocab_sizes[j], emb_sizes[j]))
            first_layer_size = first_layer_size + emb_sizes[j]
        else:
            par.add(nn.Embedding(vocab_sizes[j], opt.embed_size))
    
    if not emb_sizes:
        first_layer_size = opt.embed_size

    mod = nn.Sequential()
    mod.add_module('embeddings', par)
    mod.add_module("Join_embeddings", JoinTable(2))
    mod.add_module("Transpose", Transpose((1,2)))
    #batch_size x word_size x seqlen
    cat = ConcatTable()
    for j in range(len(kWs)):
        mod_j = nn.Sequential()
        mod_j.add_module("Conv1d", nn.Conv1d(first_layer_size, opt.num_filters, kWs[j], stride=1, padding=kWs[j]-1))
        mod_j.add_module("ReLU", ReLU())
        mod_j.add_module("Max", Max(2))
        cat.add(mod_j)

    mod.add_module("convs", cat)
    # mod.add_module("debug0", DebugLayer("debug0"))
    mod.add_module("Join_convs", JoinTable(1))

    if opt.dropout > 0:
        mod.add_module("Dropout1", nn.Dropout(opt.dropout))

    mod.add_module("Linear1", nn.Linear(len(kWs)*opt.num_filters, opt.conv_fc_layer_size))
    mod.add_module("ReLU", ReLU())

    if opt.dropout > 0:
        mod.add_module("Dropout2", nn.Dropout(opt.dropout))
    # mod.add_module("debug1", DebugLayer("debug1"))
    mod.add_module("Linear2", nn.Linear(opt.conv_fc_layer_size, nlabels))
    mod.add_module("SoftMax", SoftMax())
    return mod


def make_blstm_model(vocab_sizes, emb_sizes, nlabels, opt):
    par = ParallelTable()
    first_layer_size = 0

    for j in len(vocab_sizes):
        par.add(nn.Embedding(vocab_sizes[j], emb_sizes[j]))
        first_layer_size = first_layer_size + emb_sizes[j]
    
    mod = nn.Sequential()
    mod.add_module("embeddings", par)
    mod.add_module("Join_embeddings", JoinTable(2))

    mod.add_module("Transpose", Transpose((0,1)))
    mod.add_module("LSTM", nn.LSTM(first_layer_size, first_layer_size, 1, bidirectional=True))
    mod.add_module("Max", Max(0))

    mod.add(nn.Linear(2*first_layer_size, opt.blstm_fc_layer_size))
    mod.add(ReLU())

    if opt.dropout > 0:
        mod.add(nn.Dropout(opt.dropout))

    mod.add(nn.Linear(opt.blstm_fc_layer_size, nlabels))
    mod.add(SoftMax())

    return mod

def get_acc(model, valbatches):
    model.eval()
    correct, total = 0, 0
    for j in len(valbatches):
        sent = valbatches[j].sent
        ent_dists = valbatches[j].ent_dists
        num_dists = valbatches[j].num_dists
        labels = valbatches[j].labels
        preds = model([sent, ent_dists, num_dists])
        g_argmaxes = torch.argmax(preds, 1)
        correct = correct + labels.eq(g_argmaxes.view(-1)).sum()
        total = total + sent.size(0)
    acc = correct/total
    model.train()
    return acc

def get_multilabel_acc(model, valbatches, ignoreIdx, convens=None, lstmens=None):
    model.eval()
    if convens:
        for j in range(len(convens)):
            convens[j].eval()
    if lstmens:
        for j in range(lstmens):
            lstmens[j].eval()
    correct, total, ignored = 0, 0, 0
    pred5s, true5s = 0, 0
    nonnolabel = 0
    for j in range(len(valbatches)):
        sent = valbatches[j]["sent"].to(device)
        ent_dists = valbatches[j]["ent_dists"].to(device)
        num_dists = valbatches[j]["num_dists"].to(device)
        labels = valbatches[j]["labels"].to(device)
        labelnums = valbatches[j]["labelnums"].to(device)
        preds = None

        if convens:
            enpreds1 = convens[0]([sent, ent_dists, num_dists])
            if opt.geom:
                enpreds1.log_()
            for j in range(1, len(convens)):
                enpredsj = convens[j]([sent, ent_dists, num_dists])
                if opt.geom:
                    enpredsj.log_()
                enpreds1.add_(enpredsj)
            preds = enpreds1

        if lstmens:
            enpreds1 = lstmens[0]([sent, ent_dists, num_dists])
            if opt.geom:
                enpreds1.log_()
            for j in range(1, len(lstmens)):
                enpredsj = lstmens[j]([sent, ent_dists, num_dists])
                if opt.geom:
                    enpredsj.log_()
                enpreds1.add_(enpredsj)
            if preds:
                preds.add_(enpreds1)
            else:
                preds = enpreds1

        if not convens and not lstmens:
            preds = model([sent, ent_dists, num_dists])

        g_argmaxes = torch.argmax(preds, 1)
        nonnolabel = nonnolabel + labels[:,0].ne(ignoreIdx).sum()
        g_one_hot = torch.zeros(sent.size(0), preds.size(1)).to(device)

        numpreds = 0
        in_denominator = g_argmaxes
        for k in range(sent.size(0)):
            if not ignoreIdx or in_denominator[k] != ignoreIdx:
                g_one_hot[k].index_fill_(0, labels[k,0:labelnums[k]], 1)
                numpreds = numpreds + 1
        g_correct_buf = torch.gather(g_one_hot, 1, g_argmaxes.unsqueeze(1))
        correct = correct + g_correct_buf.sum()
        total = total + numpreds
        ignored = ignored + sent.size(0) - numpreds
    acc = correct/total
    rec = correct/nonnolabel
    logger.info("rec {}".format(rec.item()))
    logger.info("ignored {}".format(ignored/(ignored+total)))
    model.train()
    return acc, rec

def idxstostring(t, dict):
    strtbl = []
    forlimit = t.size and t.size(1) or len(t)
    for i in range(forlimit):
        strtbl.append(dict[t[i]])
    return ' '.join(strtbl)

def get_args(sent, ent_dists, num_dists, dict_):
    min_entdist = ent_dists.min()
    min_numdist = num_dists.min()
    entwrds, numwrds = [], []
    for i in range(sent.size(0)):
        if ent_dists[i]+min_entdist-1 == 0:
            entwrds.append(sent[i])
        if num_dists[i]+min_numdist-1 == 0:
            numwrds.append(sent[i])
    return idxstostring(entwrds, dict_), idxstostring(numwrds, dict_)

def eval_gens(predbatches, ignoreIdx, boxrestartidxs, convens, lstmens):
    ivocab = get_dict(opt.dict_pfx+".dict", True)
    ilabels = get_dict(opt.dict_pfx+".labels", True)
    tupfile = open(opt.preddata+"-tuples.txt", 'w')

    if ignoreIdx:
        assert ilabels[ignoreIdx] == "NONE"

    boxRestarts = None
    if boxrestartidxs:
        boxRestarts = {}
        assert boxrestartidxs.dim() == 1
        for i in range(boxrestartidxs.size(0)):
            boxRestarts[boxrestartidxs[i]] = True

    if convens:
        for j in range(len(convens)):
            convens[j].eval()

    if lstmens:
        for j in range(len(lstmens)):
            lstmens[j].eval()

    correct, total = 0, 0
    candNum = 0
    seen = {}
    ndupcorrects = 0
    nduptotal = 0
    for j in range(len(predbatches)):
        sent = predbatches[j]["sent"]
        ent_dists = predbatches[j]["ent_dists"]
        num_dists = predbatches[j]["num_dists"]
        labels = predbatches[j]["labels"]
        labelnums = predbatches[j]["labelnums"]
        preds = None

        if convens:
            enpreds1 = convens[0]([sent, ent_dists, num_dists])
            if opt.geom:
                enpreds1.log_()
            for j in range(1, len(convens)):
                enpredsj = convens[j]([sent, ent_dists, num_dists])
                if opt.geom:
                    enpredsj.log_()
                enpreds1.add_(enpredsj)
            preds = enpreds1

        if lstmens:
            enpreds1 = lstmens[0]([sent, ent_dists, num_dists])
            if opt.geom:
                enpreds1.log_()
            for j in range(1, len(lstmens)):
                enpredsj = lstmens[j]([sent, ent_dists, num_dists])
                if opt.geom:
                    enpredsj.log_()
                enpreds1.add_(enpredsj)
            
            if preds:
                preds.add_(enpreds1)
            else:
                preds = enpreds1

        g_argmaxes = torch.argmax(preds, 1)
        g_one_hot = torch.zeros(sent.size(0), preds.size(1)).to(device)
        numpreds = 0
        in_denominator = g_argmaxes
        for k in range(sent.size(0)):
            if not ignoreIdx or in_denominator[k] != ignoreIdx:
                g_one_hot[k].index_fill_(0, labels[k,0:labelnums[k]], 1)
                numpreds = numpreds + 1

        g_correct_buf = torch.gather(g_one_hot, 1, g_argmaxes.unsqueeze(1))
        
        for k in range(sent.size(0)):
            candNum = candNum + 1
            if boxRestarts and boxRestarts[candNum]:
                tupfile.write("\n")
                seen = {}
            if not ignoreIdx or in_denominator[k] != ignoreIdx:
                sentstr = idxstostring(sent[k], ivocab)
                entarg, numarg = get_args(sent[k], ent_dists[k], num_dists[k], ivocab)
                predkey = entarg + numarg + ilabels[g_argmaxes[k]]
                tupfile.write(entarg + '|' + numarg + '|' + ilabels[g_argmaxes[k]] + '\n')
                if g_correct_buf[k, 0] > 0:
                    if seen[predkey]:
                        ndupcorrects = ndupcorrects + 1
                if seen[predkey]:
                    nduptotal = nduptotal + 1
                seen[predkey] = True

        correct = correct + g_correct_buf.sum()
        total = total + numpreds

    acc = correct/total
    logger.info("prec {}".format(acc.item()))
    logger.info("nodup prec {}".format(( correct - ndupcorrects ) / ( total - nduptotal )))
    logger.info("total correct {}".format(correct.item()))
    logger.info("nodup correct {}".format(correct - ndupcorrects))
    tupfile.close()
    return acc

def set_up_saved_models():
    convens_paths = ["conv1ie-ep6-94-74.t","conv2ie-ep3-94-60.t","conv3ie-ep8-95-72.t"]
    lstmens_paths = ["blstm1ie-ep4-93-75.t", "blstm2ie-ep3-93-71.t", "blstm3ie-ep2-94-72.t"]
    opt.embed_size = 200
    opt.num_filters = 200
    opt.conv_fc_layer_size = 500
    opt.blstm_fc_layer_size = 700
    return convens_paths, lstmens_paths

#注意在前向计算的过程中张量的值不能被修改
def marginal_nll_loss(input, target, sizeAverage=True):
    loss = torch.tensor(0.0).to(device)
    for i in range(target.size(0)):
        nnz_i = target[i, -1]
        if nnz_i > 0:
            ps = input[i].index_select(0, target[i, 0:nnz_i]).sum()
            tmp = torch.log(ps+0.000001)
            loss -= tmp

    if sizeAverage:
        loss = loss/input.size(0)
    return loss


def main():
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # torch.cuda.set_device(opt.gpuid)

    trbatches, valbatches, V_sizes, nlabels, pred_batches, pboxrestartidxs = prep_data(opt.batchsize)
    emb_sizes = [opt.embed_size, opt.embed_size//2, opt.embed_size//2]
    word_pad, ent_dist_pad, num_dist_pad = V_sizes
    V_sizes = (word_pad+1, ent_dist_pad+1, num_dist_pad+1)

    if opt.just_eval:
        convens_paths, lstmens_paths = set_up_saved_models()

        convens, lstmens = None, None

        if convens_paths:
            convens = []
            for j in range(len(convens_paths)):
                mod = make_conv_model(V_sizes, emb_sizes, nlabels, opt)
                mod.load_state_dict(torch.load(convens_paths[j]))
                mod.to(device)
                convens.append(mod)
        
        if lstmens_paths:
            lstmens = []
            for j in range(len(lstmens_paths)):
                mod = make_blstm_model(V_sizes, emb_sizes, nlabels, opt)
                mod.load_state_dict(torch.load(lstmens_paths[j]))
                mod.to(device)
                lstmens.append(mod)
        eval_gens(pred_batches, opt.ignore_idx, pboxrestartidxs, convens, lstmens)
        sys.exit()

    model = None
    if opt.lstm:
        model = make_blstm_model(V_sizes, emb_sizes, nlabels, opt)
    else:
        model = make_conv_model(V_sizes, emb_sizes, nlabels, opt)

    model.to(device)

    if opt.uniform_init > 0:
        for mod in model.modules():
            if hasattr(mod, "weight"):
                nn.init.uniform_(mod.weight, -opt.uniform_init, opt.uniform_init)
            if hasattr(mod, "bias"):
                nn.init.uniform_(mod.bias, -opt.uniform_init, opt.uniform_init)

    prev_loss = float('inf')
    best_acc = 0

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    print("number of parameters", len(params))
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)#, betas=(0.9, 0.999)

    best_acc = 0
    for i in range(opt.epochs):
        logger.info("epoch {} lr {}".format(i+1, opt.lr))
        loss = torch.tensor(0.0).to(device)
        model.train()
        with torch.no_grad():
            model[0][0].weight[word_pad].zero_()
            model[0][1].weight[ent_dist_pad].zero_()
            model[0][2].weight[num_dist_pad].zero_()
        
        for j in range(len(trbatches)):#len(trbatches)
            model.zero_grad()
            # optimizer.zero_grad()
            sent = trbatches[j]["sent"].to(device)
            ent_dists = trbatches[j]["ent_dists"].to(device)
            num_dists = trbatches[j]["num_dists"].to(device)
            labels = trbatches[j]["labels"].to(device)
            preds = model([sent, ent_dists, num_dists])
            loss_ = marginal_nll_loss(preds, labels)
            loss_.backward()
            with torch.no_grad():
                loss += loss_
                if opt.lstm:
                    model[0][0].weight.grad[word_pad].zero_()
                    model[0][1].weight.grad[ent_dist_pad].zero_()
                    model[0][2].weight.grad[num_dist_pad].zero_()
                    nn.utils.clip_grad_norm_(model.parameters(), 5, 2)

                for p in model.parameters():
                    p.add_(-opt.lr*p.grad)

                # optimizer.step()
                model[0][0].weight[word_pad].zero_()
                model[0][1].weight[ent_dist_pad].zero_()
                model[0][2].weight[num_dist_pad].zero_()

        logger.info("train loss:{}".format(loss.item()/len(trbatches)))
        acc, rec = get_multilabel_acc(model, valbatches, opt.ignore_idx)
        logger.info("acc:{}".format(acc.item()))

        savefi = "{}-ep{}-{}-{}.pt".format(opt.savefile, i, acc.item(), rec.item())
        logger.info("saving to {}".format(savefi))
        torch.save(model.state_dict(), savefi)
        
        valloss = -acc
        if valloss >= prev_loss and opt.lr > 0.0001:
            opt.lr = opt.lr*opt.lr_decay
        prev_loss = valloss

if __name__ == "__main__":
    main()


