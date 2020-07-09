import sys
import os
import re

folder = sys.argv[1]

files = os.listdir(folder)

for f in  files:
    if ".h5-tuples.txt" in f:
        fp = open(os.path.join(folder, f), "r")
        text = fp.read()
        text = re.sub(r"[^\n]+NEXT-DAYNAME\n", "", text)
        fp.close()
        fp = open(os.path.join(folder, f), "w")
        fp.write(text)
        fp.close()































