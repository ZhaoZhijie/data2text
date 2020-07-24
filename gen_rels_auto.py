from extractor import main, get_dict
import sys
import os
from logger import logger
import re

def get_ignore_idx(exp):
    dataset = "D2"
    label_path = os.path.join("data", dataset, dataset+".labels")
    dict_, _ = get_dict(label_path)
    return dict_["NONE"]

def get_datafile_path(exp):
    dataset = "D2"
    return os.path.join("data", dataset, dataset+".h5")

def get_sys(exp):
    return exp[0:2]

def get_dict_pfx(exp):
    dataset = "D2"
    return os.path.join("data", dataset, dataset)

def get_eval_models(exp):
    dataset = "D2"
    return os.path.join("eval_models", dataset, "best","use")

def generate_h5(seed, exp, step, test=False, avg=False):
    gen_folder = "test" if test else "valid"
    midstr = "_avg" if avg else ""
    gen_fi = "experiments/exp-seed-{}/exp-{}/gens/{}/predictions{}_{}.txt".format(seed, exp, gen_folder, midstr, step)
    dict_pfx = "data/D2/D2"
    output_fi = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, midstr, step)
    input_path = "data/D2"
    test_tag = "-test" if test else ""
    cmd = 'python data_utils_ex.py -mode prep_gen_data -gen_fi "{}" -dict_pfx "{}" -output_fi "{}" -input_path "{}" {}'\
            .format(gen_fi, dict_pfx, output_fi, input_path, test_tag)
    os.system(cmd)
    return os.path.exists(output_fi)

def remove_h5(seed, exp, step, test=False, avg=False):
    gen_folder = "test" if test else "valid"
    midstr = "_avg" if avg else ""
    output_fi = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, midstr, step)
    if os.path.exists(output_fi):
        os.remove(output_fi)

def get_model_steps(file):
    steps = re.findall(r'_([0-9]+).txt', file)
    if not steps:
        return None
    return int(steps[0])

def get_predictions(seed, exp, test):
    gens = "test" if test else "valid"
    pred_folder = "experiments/exp-seed-{}/exp-{}/gens/{}".format(seed, exp, gens)
    files = os.listdir(pred_folder)
    step_files = []
    for f in files:
        step_files.append((get_model_steps(f), os.path.join(pred_folder, f)))
    def get_first(elem):
        return elem[0]
    step_files.sort(key=get_first)
    return [tup[1] for tup in step_files]

def get_seeds(seedstr):
    return [int(seed) for seed in seedstr.split(",")]

def gen_rels(seeds, test=False):
    exps = ["S1D1", "S1D2", "S4D1", "S4D2"]
    gen_folder = "test" if test else "valid"
    for seed in seeds:
        for exp in exps:
            sysname = get_sys(exp)
            avg =  sysname == "S1"
            midstr = "_avg" if avg else ""
            datafile = get_datafile_path(exp)
            dict_pfx = get_dict_pfx(exp)
            ignore_idx = get_ignore_idx(exp)
            eval_models = get_eval_models(exp)
            pred_files = get_predictions(seed, exp, test)
            for pred_file in pred_files:
                step = get_model_steps(pred_file)
                preddata = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, midstr, step)
                tupfile = preddata+"-tuples.txt"
                if os.path.exists(tupfile):
                    logger.info("tuple file already exists:{}".format(tupfile))
                    continue
                sys.argv = [sys.argv[0], "-datafile", datafile, "-preddata", preddata, "-dict_pfx", dict_pfx, "-ignore_idx", str(ignore_idx), "-eval_models", eval_models, "-just_eval"]
                if test:
                    sys.argv.append("-test")
                gtag = generate_h5(seed, exp, step, test, avg)
                if gtag:
                    main()
                    remove_h5(seed, exp, step, test, avg)
                else:
                    logger.info("gen h5 file error, seed:{} exp:{} step:{} test:{}".format(seed, exp, step, test))


if __name__ == "__main__":
    seeds = get_seeds(sys.argv[1])
    test = sys.argv[2] == "True"
    gen_rels(seeds, test)





