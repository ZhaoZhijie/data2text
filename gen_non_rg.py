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


def get_model_steps(file):
    steps = re.findall(r'_([0-9]+).h5', file)
    if not steps:
        return None
    return int(steps[0])

def get_tuple_files(seed, exp, test):
    gens = "test_ex" if test else "valid_ex"
    tuple_folder = "experiments/exp-seed-{}/exp-{}/gens/{}".format(seed, exp, gens)
    files = os.listdir(tuple_folder)
    step_files = []
    for f in files:
        if "tuples" in f:
            step_files.append((get_model_steps(f), os.path.join(tuple_folder, f)))
    def get_first(elem):
        return elem[0]
    step_files.sort(key=get_first)
    return [tup[1] for tup in step_files]


seed = sys.argv[1]
exp = sys.argv[2]
step_start = int(sys.argv[3])
step_end = int(sys.argv[4])
test = sys.argv[5] == "True"

gen_folder = "test" if test else "valid"
avg =  "_avg" if get_sys(exp) == "S1" else ""
gold_tuples = get_gold_tuples(exp, test)


tup_files = get_tuple_files(seed, exp, test)
for f in tup_files:
    try:
        step = get_model_steps(f)
        predtuples = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5-tuples.txt".format(seed, exp, gen_folder, avg, step)
        print("current predtuples", predtuples)
        calc_precrec(gold_tuples, predtuples)
        calc_dld(gold_tuples, predtuples)
    except Exception as e:
        print(e)


































