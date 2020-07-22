import os
import sys
import re



def get_sys(exp):
    return exp[0:2]

def get_dataset(exp):
    return exp[2:4]

def get_seeds(seedstr):
    return [int(seed) for seed in seedstr.split(",")]

def get_model_steps(file):
    steps = re.findall(r'_([0-9]+).txt', file)
    if not steps:
        return None
    return int(steps[0])

def get_predictions(seed, exp, test):
    gens = "test" if test else "valid"
    pred_folder = "experiments/exp-seed-{}/exp-{}/gens/{}".format(seed, exp, gens)
    files = os.listdir(pred_folder)
    step_files = [(get_model_steps(file), file) for file in files]
    def get_first(elem):
        return elem[0]
    step_files.sort(key=get_first)
    return [tup[1] for tup in step_files]

def get_golden_output(exp, test):
    sysname = get_sys(exp)
    dataset = get_dataset(exp)
    mid = "test" if test else "validation"
    pre = "S4" if sysname == "S4" else ""
    return "data/{}{}_{}_txt.txt".format(pre, dataset, mid)



if __name__ == "__main__":
    seeds = get_seeds(sys.argv[1])
    test = sys.argv[2] == "True"
    exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    rcd_name_post = "_test" if test else ""
    for seed in seeds:
        for exp in exps:
            rcd_file = "logs/exp-seed-{}/bleu_score_{}{}.txt".format(seed, exp, rcd_name_post)
            wf = open(rcd_file, "w")
            pred_files = get_predictions(seed, exp, test)
            for pred_file in pred_files:
                print("current", pred_file)
                cmd = "cat {} | sacrebleu --force {}".format(pred_file, get_golden_output(exp, test))
                res = os.popen(cmd)
                wf.write(pred_files+"\n"+res+"\n")
            wf.close()














































