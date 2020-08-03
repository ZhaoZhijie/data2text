#!/usr/bin/env python
from onmt.bin.translate import translate, _get_parser
import sys
from cp import scp_files
import os
from logger import logger
import torch
import re
from exp_utils import prepare_common_model

exps = ["S1D1","S1D2","S4D1","S4D2"]


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

def prepare_avg_model(seed, exp, n, last=False):
    max_n = n
    min_n = n - 4
    if min_n < 1:
        min_n = 1
    src_models = []
    tar_folder = "experiments/exp-seed-{}/exp-{}/models/".format(seed, exp)
    src_folder = "/home/zzjstars/zj17501_disk/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
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

def get_steps_no(nostr):
    nos = nostr.split(",")
    return [int(no) for no in nos]

def get_best_bleu(seed, exp):
    fpath = "logs/exp-seed-{}/bleu_score_{}.txt".format(seed, exp)
    f = open(fpath, "r")
    texts = f.readlines()
    step = 0
    step_bleu = []
    for text in texts:
        if "predictions" in text:
            step = int(re.findall(r"_([0-9]+).txt", text)[0])
        elif "BLEU" in text:
            bleu = float(re.findall(r"version.1.4.12 = ([0-9]+.[0-9]+)", text)[0])
            step_bleu.append((bleu, step))
    def get_first(elem):
        return elem[0]
    step_bleu.sort(key=get_first, reverse=True)
    f.close()
    return step_bleu[0][1]

def get_best_RG(seed, exp):
    fpath = "logs/exp-seed-{}/gen-rels.out".format(seed)
    f = open(fpath, "r")
    texts = f.readlines()
    isexp = False
    step = 0
    step_rg = []
    for text in texts:
        if "predictions" in text:
            if exp in text:
                step = int(re.findall(r"_([0-9]+).h5", text)[0])
                isexp = True
            else:
                isexp = False
        elif isexp and "nodup prec" in text:
            prec = float(re.findall(r"nodup prec ([0-9]+.[0-9]+)", text)[0])
            step_rg.append((prec, step))
    def get_first(elem):
        return elem[0]
    step_rg.sort(key=get_first, reverse=True)
    f.close()
    return step_rg[0][1]

def get_best_other(seed, exp):
    fpath = "logs/exp-seed-{}/gen-metrics.out".format(seed)
    f = open(fpath, "r")
    texts = f.readlines()
    isexp = False
    step = 0
    prec = 0
    rec = 0
    co = 0
    step_other = []
    for text in texts:
        if "predictions" in text:
            if exp in text:
                step = int(re.findall(r"_([0-9]+).h5", text)[0])
                isexp = True
            else:
                isexp = False
        elif isexp and "prec" in text and "rec" in text:
            prec = float(re.findall(r"prec: ([0-9]+.[0-9]+)", text)[0])
            rec = float(re.findall(r" rec: ([0-9]+.[0-9]+)", text)[0])
        elif isexp and "avg score" in text:
            co = float(re.findall(r"avg score: ([0-9]+.[0-9]+)", text)[0])
            step_other.append((prec, rec, co, step))
    def get_first(elem):
        return elem[0]
    def get_second(elem):
        return elem[1]
    def get_third(elem):
        return elem[2]
    step_other.sort(key=get_first, reverse=True)
    best_prec_step = step_other[0][3]
    step_other.sort(key=get_second, reverse=True)
    best_rec_step = step_other[0][3]
    step_other.sort(key=get_third, reverse=True)
    best_co_step = step_other[0][3]
    f.close()
    return best_prec_step, best_rec_step, best_co_step


def get_best_models(seed):
    #get the steps of best bleu score, RG，CS，CO
    best_models = {}
    for exp in exps:
        bleu_step = get_best_bleu(seed, exp)
        rg_step = get_best_RG(seed, exp)
        cs_prec_step, cs_rec_step, co_step = get_best_other(seed, exp)
        logger.info("seed:{} exp:{} bleu_step:{} rg_step:{} cs_prec_step:{} cs_rec_step:{} co_step:{}".format(seed, exp, bleu_step, rg_step, cs_prec_step, cs_rec_step, co_step))
        best_models[exp] = set([bleu_step, rg_step, cs_prec_step, cs_rec_step, co_step])
    return best_models

def get_exps(expstr):
    return [exp for exp in expstr.split(",")]

if __name__ == "__main__":
    seed = int(sys.argv[1])
    if len(sys.argv) >= 3:
        exps = get_exps(sys.argv[2])
    else:
        exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    best_models = get_best_models(seed)
    for exp in exps:
        sys.argv = [sys.argv[0], "--config", "config-seed-{}/translate_{}.cfg".format(seed, exp)]
        avg = exp[0:2] == "S1"
        midstr = "_avg" if avg else ""
        models_steps = best_models[exp]
        for step in models_steps:
            output = "experiments/exp-seed-{}/exp-{}/gens/test/predictions{}_{}.txt".format(seed, exp, midstr, step)
            if os.path.exists(output):
                logger.info("prediction already done for {}".format(output))
                continue
            parser = _get_parser()
            opt = parser.parse_args()
            opt.output = output
            opt.src = "data/{}_test_data.txt".format(exp if "S4" in exp else exp[2:4])
            opt.models = ["experiments/exp-seed-{}/exp-{}/models/model{}_step_{}.pt".format(seed, exp, midstr, step)]
            opt.log_file = "experiments/exp-seed-{}/exp-{}/translation{}-test-log.txt".format(seed, exp, midstr, step)
            tag = prepare_model(seed, exp, step//1000, avg, True)
            if tag:
                translate(opt)
                clear_translate_model(seed, exp, step//1000, avg)
            else:
                logger.info("translate error step={}".format(step)) 




