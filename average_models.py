from average_checkpoints import main
import sys

def get_avg_args(exp, steps):
    args = []
    args.append("--folder")
    args.append("exp-{}".format(exp))
    args.append("--output")
    args.append("avg_model_{}.pt".format(steps*1000))
    args.append("--steps")
    start = steps - 4
    if start < 1:
        start = 1
    end = steps + 1
    for i in range(start, end):
        args.append(str(i*1000))
    return args
    

if __name__ == "__main__":
    exp = sys.argv[1]
    max_steps = int(sys.argv[2])
    sys.argv[0] = "average_checkpoints.py"
    for i in range(1, max_steps+1):
        sys.argv = [sys.argv[0]] + get_avg_args(exp, i)
        main()



















