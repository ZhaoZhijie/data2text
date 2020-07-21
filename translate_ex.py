#!/usr/bin/env python
from onmt.bin.translate import translate, _get_parser
import sys
from cp import scp_files
import os
from logger import logger
import torch

def get_avg_args(seed, exp, n):
    args = []
    args.append("--folder")
    args.append("exp-seed-{}/exp-{}".format(seed, exp))
    args.append("--output")
    args.append("avg_model_{}.pt".format(n*1000))
    args.append("--steps")
    start = n - 4
    if start < 1:
        start = 1
    end = n + 1
    for i in range(start, end):
        args.append(str(i*1000))
    return args

def prepare_model(seed, exp, n, avg=False, last=False):
    if avg:
        return prepare_avg_model(seed, exp, n, last)
    else:
        return prepare_common_model(seed, exp, n)

def prepare_common_model(seed, exp, n):
    src = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/model_step_{}.pt".format(seed, exp, n*1000)
    tar_folder = "experiments/exp-seed-{}/exp-{}/models/".format(seed, exp)
    logger.info("common model {}".format(src))
    succs, fails = scp_files(src, tar_folder, get=True)
    if len(succs) != 1:
        return False
    return True

def prepare_avg_model(seed, exp, n, last=False):
    max_n = n
    min_n = n - 4
    if min_n < 1:
        min_n = 1
    src_models = []
    tar_folder = "experiments/exp-seed-{}/exp-{}/models/".format(seed, exp)
    src_folder = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
    merged = []
    for i in range(min_n, max_n+1):
        steps = i*1000
        model = "model_step_{}.pt".format(i*1000)
        tar = os.path.join(tar_folder, model)
        merged.append(tar)
        if not os.path.exists(tar):
            src = os.path.join(src_folder, model)
            src_models.append(src)
    logger.info("{} merged files {}".format(n, merged))
    succs, fails = scp_files(src_models, tar_folder, get=True)
    if len(succs) != len(src_models):
        return False
    avg_cp = average_checkpoints(merged)
    output = "model_avg_step_{}.pt".format(n*1000)
    torch.save(avg_cp, os.path.join(tar_folder, output))
    if last:
        for model in merged:
            os.remove(model)
    else:
        if len(merged) >= 5:
            os.remove(merged[0])
    return True

def clear_translate_model(seed, exp, n, avg=False):
    midstr = "_avg" if avg else ""
    path = "experiments/exp-seed-{}/exp-{}/models/model{}_step_{}.pt".format(seed, exp, midstr, n*1000)
    if os.path.exists(path):
        os.remove(path)

def average_checkpoints(checkpoint_files):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        m = torch.load(checkpoint_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']
        
        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)
                
            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)
    
    return {"vocab": vocab, 'opt': opt, 'optim': None,
            "generator": avg_generator, "model": avg_model}


if __name__ == "__main__":
    seed = int(sys.argv[1])
    exp = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    avg = sys.argv[5] == "True"
    test = sys.argv[6] == "True"
    sys.argv = [sys.argv[0], "--config", "config-seed-{}/translate_{}.cfg".format(seed, exp)]
    logger.info("seed-{} exp-{} start-{} end-{}".format(seed, exp, start, end))
    midstr = "_avg" if avg else ""
    gens = "test" if test else "valid"
    for i in range(start, end+1):
        steps = i*1000
        parser = _get_parser()
        opt = parser.parse_args()
        opt.output = "experiments/exp-seed-{}/exp-{}/gens/{}/predictions{}_{}.txt".format(seed, exp, gens, midstr, steps)
        opt.models = ["experiments/exp-seed-{}/exp-{}/models/model{}_step_{}.pt".format(seed, exp, midstr, steps)]
        opt.log_file = "experiments/exp-seed-{}/exp-{}/translation{}-log.txt".format(seed, exp, midstr, steps)
        tag = prepare_model(seed, exp, i, avg, i==end)
        if tag:
            translate(opt)
            clear_translate_model(seed, exp, i, avg)
        else:
            logger.info("translate error n={}".format(i))



