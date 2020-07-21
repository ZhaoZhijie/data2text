from cp import put_files
import os
import time
from logger import logger




def cp_models_generated(seeds=[]):
    exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    for seed in seeds:
        for exp in exps:
            folder = "experiments/exp-seed-{}/exp-{}/models".format(seed, exp)
            savepath = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
            files = os.listdir(folder)
            if files:
                filepaths = [os.path.join(folder, file) for file in files]
                succs, fails = put_files(filepaths, savepath)
                for succ in succs:
                    os.remove(succ)
                    logger.info("移除模型文件{}".format(succ))
                if fails:
                    logger.info("传输失败的文件{}".format(fails))

def monitor(seeds):
    for i in range(200000):
        logger.info("check models folders")
        cp_models_generated(seeds)
        time.sleep(30)

def get_seeds(seedstr):
    return [int(seed) for seed in seedstr.split(",")]

seeds = get_seeds(sys.argv[1])
monitor(seeds)





























