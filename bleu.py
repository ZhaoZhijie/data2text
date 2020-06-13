import os
import sys


if __name__ == "__main__":
    exp = sys.argv[1]
    max_steps = int(sys.argv[2])
    for i in range(1, max_steps+1):
        steps = i*1000
        cmd = "cat experiments/exp-{}/gens/valid/predictions_avg_{}.txt | sacrebleu --force data2/valid_output.txt>>bleu_score_exp_{}_avg.txt".format(exp, steps, exp)
        os.system(cmd)















