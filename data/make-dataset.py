import json
import argparse
import os

DELIM = u"￨"
MISSING = "N/A"


def check_replace(src, tgt):
    newstr = src.replace(" ","_")
    if src in tgt:
        tgt = tgt.replace(src, newstr)
    return tgt

def parse_data(data):
    source = []
    target = []
    for i in range(len(data)):
        obj = data[i]
        src_attrs = []
        tgt_words = obj["summary"]
        tgt_str = " ".join(tgt_words)
        #处理day
        day = obj["day"]
        if day != MISSING:
            day_units = day.split("_")
            src_attrs.append(str(int(day_units[2]))+"￨day￨global￨year")
            src_attrs.append(str(int(day_units[0]))+"￨day￨global￨month")
            src_attrs.append(str(int(day_units[1]))+"￨day￨global￨date")
        #处理line字段
        lines = ["home_line", "vis_line"]
        for l in range(len(lines)):
            line_name = lines[l]
            line_obj = obj[line_name]
            for key,val in line_obj.items():
                if val != MISSING:
                    tgt_str = check_replace(val, tgt_str)
                    val = val.replace(" ", "_")
                    src_attrs.append("{}￨line￨{}￨{}".format(val, line_name, key))
        #处理box_score
        box_score = obj["box_score"]
        filters = ["PLAYER_NAME"]
        for key in box_score.keys():
            if key not in filters:
                score_map = box_score[key]
                for no, val in score_map.items():
                    if val != MISSING:
                        if val == obj["home_city"]:
                            val == "home"
                        elif val == obj["vis_city"]:
                            val == "vis"
                        tgt_str = check_replace(val, tgt_str)
                        val = val.replace(" ","_")
                        src_attrs.append("{}￨box_score￨{}￨{}".format(val, no, key))
        src_str = " ".join(src_attrs)
        source.append(src_str)
        target.append(tgt_str)
    return source, target

def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', dest='folder', required=True,
                        help='Save the preprocessed dataset to this folder')
    parser.add_argument('--keep-na', dest='keep_na', action='store_true',
                        help='Activate to keep NA in the dataset')

    args = parser.parse_args()
    names = ["train", "valid", "test"]
    for name in names:
        datapath = f'data/rotowire/{name}.json'
        with open(datapath, "r", encoding="utf-8") as f:
            txt = f.read()
            dataset = json.loads(txt)
            source, target = parse_data(dataset)
            path_input = os.path.join(args.folder, f'{name}_input.txt')
            path_output = os.path.join(args.folder, f'{name}_output.txt')
            save_data("\n".join(source), path_input)
            save_data("\n".join(target), path_output)












