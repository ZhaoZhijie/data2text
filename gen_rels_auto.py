from extractor import main, get_dict
import sys
import os

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

def generate_h5(seed, exp, n, test=False, avg=False):
    gen_folder = "test" if test else "valid"
    midstr = "_avg" if avg else ""
    gen_fi = "experiments/exp-seed-{}/exp-{}/gens/{}/predictions{}_{}.txt".format(seed, exp, gen_folder, midstr, n*1000)
    dict_pfx = "data/D2/D2"
    output_fi = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, midstr, n*1000)
    input_path = "data/D2"
    cmd = 'python data_utils_ex.py -mode prep_gen_data -gen_fi "{}" -dict_pfx "{}" -output_fi "{}" -input_path "{}" {}'\
            .format(gen_fi, dict_pfx, output_fi, input_path, test)
    os.system(cmd)
    print("gen_fi", gen_fi)
    print("output_fi", output_fi)
    return os.path.exists(output_fi)

def remove_h5(seed, exp, n, test=False, avg=False):
    gen_folder = "test" if test else "valid"
    midstr = "_avg" if avg else ""
    output_fi = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, midstr, n*1000)
    if os.path.exists(output_fi):
        os.remove(output_fi)

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
            step_start = 1
            step_end = 35 if sysname == "S1" else 20
            datafile = get_datafile_path(exp)
            dict_pfx = get_dict_pfx(exp)
            ignore_idx = get_ignore_idx(exp)
            eval_models = get_eval_models(exp)
            for i in range(step_start, step_end+1):
                preddata = "experiments/exp-seed-{}/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(seed, exp, gen_folder, avg, i*1000)
                sys.argv = [sys.argv[0], "-datafile", datafile, "-preddata", preddata, "-dict_pfx", dict_pfx, "-ignore_idx", str(ignore_idx), "-eval_models", eval_models, "-just_eval"]
                if test:
                    sys.argv.append("-test")
                generate_h5(seed, exp, i, test, avg)
                main()
                remove_h5(seed, exp, i, test)


if __name__ == "__main__":
    seeds = get_seeds(sys.argv[1])
    test = sys.argv[2] == "True"
    gen_rels(seeds, test)





