import os
import re
import argparse
from logger import logger
from cp import scp_files

exps = ["S1D1", "S1D2", "S4D1", "S4D2"]

def make_experiments_dirs(seeds):
    for seed in seeds:
        for exp in exps:
            exp_folder = "experiments/exp-seed-{}/exp-{}/".format(seed, exp)
            gens_folder = os.path.join(exp_folder, "gens")
            items = ["valid", "valid_ex", "test", "test_ex"]
            for item in items:
                path = os.path.join(gens_folder, item)
                if not os.path.exists(path):
                    os.makedirs(os.path.join(gens_folder, item))
            models_folder = os.path.join(exp_folder, "models")
            os.makedirs(models_folder)
            data_folder = os.path.join(exp_folder, "data")
            os.makedirs(data_folder)
            os.system("cp -r experiments/exp-seed-123/exp-{}/data/* experiments/exp-seed-{}/exp-{}/data".format(exp, seed, exp))

def make_log_dirs(seeds):
    for seed in seeds:
        path = "logs/exp-seed-{}".format(seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
def make_cfg_dirs(seeds):
    for seed in seeds:
        cfg_folder = "config-seed-{}".format(seed)
        if not os.path.exists(cfg_folder):
            os.makedirs(cfg_folder)
            os.system("cp -r config-seed-123/* config-seed-{}".format(seed))

def update_cfg(seeds):
    for seed in seeds:
        folder = "config-seed-{}".format(seed)
        files = os.listdir(folder)
        for f in files:
            path = os.path.join(folder, f)
            with open(path, "r") as fi:
                text = fi.read()
                if os.path.splitext(f)[-1] == ".sh":
                    text = text.replace("123", seed)
                elif "train_" in f:
                    text = text.replace("seed-123", "seed-{}".format(seed))
                    text = text.replace("seed: 123", "seed: {}".format(seed))
                    text = text.replace("\ntrain_from", "\n#train_from")
                fi.close()
                with open(path, "w") as fo:
                    fo.write(text)
                    fo.close()


def update_continued_train_cfg(seeds):
    for seed in seeds:
        folder = "config-seed-{}".format(seed)
        files = os.listdir(folder)
        for f in files:
            if "train_" in f:
                path = os.path.join(folder, f)
                exp = re.findall(r"S[0-9]D[0-9]", f)[0]
                step = get_last_step(seed, exp)
                max_step = 36000 if "S1" in exp else 27000
                with open(path, "r") as fi:
                    text = fi.read()
                    text = text.replace("\n#train_from", "\ntrain_from")
                    text = re.sub(r"model_step_[0-9]+.pt", "model_step_{}.pt".format(step), text)
                    text = re.sub(r"train_steps: [0-9]+", "train_steps: {}".format(max_step), text)
                    fi.close()
                    with open(path, "w") as fo:
                        fo.write(text)
                        fo.close()


def make_dirs(seeds):
    make_experiments_dirs(seeds)
    make_log_dirs(seeds)
    make_cfg_dirs(seeds)
    update_cfg(seeds)


def get_seeds(seedstr):
    return [seed for seed in seedstr.split(",")]

def get_exps(expstr):
    return [exp for exp in expstr.split(",")]

def get_continued_models(seeds, exps):
    fw = open("continued_models.txt", "a")
    for seed in seeds:
        for exp in exps:
            step = get_last_step(seed, exp)
            tag = prepare_common_model(seed, exp, step//1000)
            path = "experiments/exp-seed-{}/exp-{}/models/model_step_{}.pt".format(seed, exp, step)
            fw.write(path+"\n")
    fw.close()



def prepare_common_model(seed, exp, n):
    src = "/home/zzjstars/zj17501_disk/zjmodels/exp-seed-{}/exp-{}/models/model_step_{}.pt".format(seed, exp, n*1000)
    tar_folder = "experiments/exp-seed-{}/exp-{}/models/".format(seed, exp)
    logger.info("common model {}".format(src))
    succs, fails = scp_files(src, tar_folder, get=True)
    if len(succs) != 1:
        return False
    return True



def get_last_step(seed, exp):
    log_folder = "logs/exp-seed-{}/".format(seed)
    files = os.listdir(log_folder)
    steps = []
    for f in files:
        if exp in f:
            fpath = os.path.join(log_folder, f)
            fp = open(fpath, "r")
            text = fp.read()
            steps += re.findall(r"Saving checkpoint experiments/exp-seed-[0-9]+/exp-S[0-9]D[0-9]/models/model_step_([0-9]+).pt",text)
            fp.close()
    step = 0
    if steps:
        steps = [int(step) for step in steps]
        steps.sort()
        step = steps[-1]
        # if step > 18000 and "S4" in exp:
        #     step = 0
        # if step > 23000 and "S1" in exp:
        #     step = 0
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='experiments utils')
    parser.add_argument('-seeds', type=str, default="",
                        help="random seeds for experiments to be config")
    parser.add_argument('-type', type=str, default="", choices=['mkdirs', 'update_train', "contnued"],
                        help="action type")
    parser.add_argument('-exps', type=str, default="",
                        help="experiments name")
    args = parser.parse_args()

    seeds = get_seeds(args.seeds)
    if args.type == "mkdirs":
        make_dirs(seeds)
    elif args.type == "update_train":
        update_continued_train_cfg(seeds)
    elif args.type == "contnued":
        if args.exps:
            exps = get_exps(args.exps)
        get_continued_models(seeds, exps)
    else:
        print("action type error")













            

































