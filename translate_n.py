#!/usr/bin/env python
from onmt.bin.translate import translate, _get_parser
import sys

if __name__ == "__main__":
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    exp = sys.argv[1]
    sys.argv = [sys.argv[0], "--config", "translate{}.cfg".format(exp)]
    print("exp-{} start-{} end-{}".format(exp, start, end))
    for steps in range(start, end+1):
        steps *= 1000
        parser = _get_parser()
        opt = parser.parse_args()
        opt.output = "experiments/exp-{}/gens/valid/predictions_{}.txt".format(exp, steps)
        opt.models = ["experiments/exp-{}/models/model_step_{}.pt".format(exp, steps)]
        opt.log_file = "experiments/exp-{}/translation-{}-log.txt".format(exp, steps)
        translate(opt)




























