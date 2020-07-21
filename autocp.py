from cp import scp_files
import os
import time


def cp_models_generated(seeds=[]):
    exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    for seed in seeds:
        for exp in exps:
            folder = "experiments/exp-seed-{}/exp-{}/models".format(seed, exp)
            savepath = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
            files = os.listdir(folder)
            if files:
                filepaths = [os.path.join(folder, file) for file in files]
                succs, fails = scp_files(filepaths, savepath)
                for succ in succs:
                    os.remove(succ)
                    print("移除模型文件{}".format(succ))
                if fails:
                    print("传输失败的文件{}".format(fails))

def monitor(seeds):
    for i in range(200000):
        print("检测是否有模型生成")
        cp_models_generated(seeds)
        time.sleep(30)

seeds = [287, 684, 963]
monitor(seeds)





























