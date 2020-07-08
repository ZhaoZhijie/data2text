import os
import sys
import argparse



parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-input_path', type=str, default="",
                    help="path to input")
parser.add_argument('-gen_path', type=str, default="",
                    help="path to generated texts")
parser.add_argument('-h5_path', type=str, default="",
                    help="path to generated h5 files")
parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                    help="prefix of .dict and .labels files")
parser.add_argument('-test', action='store_true', help='use test data')

opt = parser.parse_args()

dict_pfx = opt.dict_pfx
input_path = opt.input_path
test = "-test" if opt.test else ""

gen_files= os.listdir(opt.gen_path)
for f in gen_files:
    gen_fi = os.path.join(opt.gen_path, f)
    h5_name = f.replace(".txt", ".h5")
    output_fi = os.path.join(opt.h5_path, h5_name)
    cmd = 'python data_utils.py -mode prep_gen_data -gen_fi "{}" -dict_pfx "{}" -output_fi "{}" -input_path "{}" {}'\
        .format(gen_fi, dict_pfx, output_fi, input_path, test)
    os.system(cmd)
    

















































