from extractor import main, get_dict
import sys
import os

def get_ignore_idx(exp):
    dataset = exp[2:4]
    label_path = os.path.join("data", dataset, dataset+".labels")
    dict_, _ = get_dict(label_path)
    return dict_["NONE"]

def get_datafile_path(exp):
    dataset = exp[2:4]
    return os.path.join("data", dataset, dataset+".h5")

def get_sys(exp):
    return exp[0:2]

def get_dict_pfx(exp):
    dataset = exp[2:4]
    return os.path.join("data", dataset, dataset)

def get_eval_models(exp):
    dataset = exp[2:4]
    return os.path.join("eval_models", dataset, "best","use")


exp = sys.argv[1]
step_start = int(sys.argv[2])
step_end = int(sys.argv[3])
test = True if sys.argv[4] == "True" else False

gen_folder = "test" if test else "valid"
avg =  "_avg" if get_sys(exp) == "S1" else ""

datafile = get_datafile_path(exp)
dict_pfx = get_dict_pfx(exp)
ignore_idx = get_ignore_idx(exp)
eval_models = get_eval_models(exp)


for i in range(step_start, step_end+1):
    step = i*1000
    preddata = "experiments/exp-{}/gens/{}_ex/predictions{}_{}.h5".format(exp, gen_folder, avg, step)
    sys.argv = [sys.argv[0], "-datafile", datafile, "-preddata", preddata, "-dict_pfx", dict_pfx, "-ignore_idx", str(ignore_idx), "-eval_models", eval_models, "-just_eval"]
    # print("step {} preddata {}".format(step, preddata))
    if test:
        sys.argv.append("-test")
    main()



