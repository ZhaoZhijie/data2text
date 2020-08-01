from cp import scp_files
import os
import time
from logger import logger
import sys




def cp_models_generated(seeds=[]):
    exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    continued_file = "continued_models.txt"
    continued = ""
    if os.path.exists(continued_file):
        fp = open(continued_file, "r")
        continued = fp.read()
    for seed in seeds:
        for exp in exps:
            folder = "experiments/exp-seed-{}/exp-{}/models".format(seed, exp)
            savepath = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
            files = os.listdir(folder)
            if files:
                filepaths = [os.path.join(folder, file) for file in files]
                scppaths = []
                for path in filepaths:
                    if path not in continued:
                        scppaths.append(path)
                succs, fails = scp_files(scppaths, savepath)
                for succ in succs:
                    os.remove(succ)
                    logger.info("remove model file{}".format(succ))
                if fails:
                    logger.info("copy failed {}".format(fails))

def monitor(seeds):
    for i in range(200000):
        logger.info("check models folders")
        cp_models_generated(seeds)
        time.sleep(30)

def get_seeds(seedstr):
    return [int(seed) for seed in seedstr.split(",")]

seeds = get_seeds(sys.argv[1])
monitor(seeds)





























