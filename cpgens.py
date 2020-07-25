import re
import os


seeds = [123, 528, 287, 684, 963]
exps = ["S1D1", "S1D2", "S4D1", "S4D2"]

tar_folder = "genfiles"

for seed in seeds:
    for exp in exps:
        gen_folder = "experiments/exp-seed-{}/exp-{}/gens/*".format(seed, exp)
        copy_foler = "{}/experiments/exp-seed-{}/exp-{}/gens/".format(tar_folder, seed, exp)
        os.makedirs(copy_foler)
        cmd = "cp -r {} {}".format(gen_folder, copy_foler)
        os.system(cmd)




































