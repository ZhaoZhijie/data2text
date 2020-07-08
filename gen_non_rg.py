import sys
import os
from non_rg_metrics import calc_precrec, calc_dld

def get_sys(exp):
    return exp[0:2]

def get_dataset(exp):
    return exp[2:4]

def get_gold_tuples(exp, test):
    dataset = get_dataset(exp)
    return "data/{}/{}_validation_txt.h5-tuples.txt".format(dataset, dataset)

exp = sys.argv[1]
step_start = int(sys.argv[2])
step_end = int(sys.argv[3])
test = True if sys.argv[4] == "True" else False

gen_folder = "test" if test else "valid"
avg =  "_avg" if get_sys(exp) == "S1" else ""
gold_tuples = get_gold_tuples(exp, test)


for i in range(step_start, step_end+1):
    try:
        step = i*1000
        predtuples = "experiments/exp-{}/gens/{}_ex/predictions{}_{}.h5-tuples.txt".format(exp, gen_folder, avg, step)
        print("current predtuples", predtuples)
        calc_precrec(gold_tuples, predtuples)
        calc_dld(gold_tuples, predtuples)
    except Exception as e:
        print(e)

































