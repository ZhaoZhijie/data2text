#!/usr/bin/env python
from onmt.bin.translate import translate, _get_parser
import sys

if __name__ == "__main__":
    max_steps = int(sys.argv[2])
    exp = sys.argv[1]
    sys.argv = [sys.argv[0], "--config", "translate{}.cfg".format(exp)]
    print("exp-{} max_steps-{}".format(exp, max_steps))
    for steps in range(1, max_steps+1):
        steps *= 1000
        parser = _get_parser()
        opt = parser.parse_args()
        opt.output = "experiments/exp-{}/gens/valid/predictions_avg_{}.txt".format(exp, steps)
        opt.models = ["experiments/exp-{}/models/avg_model_{}.pt".format(exp, steps)]
        opt.log_file = "experiments/exp-{}/translation-avg-{}-log.txt".format(exp, steps)
        translate(opt)




























