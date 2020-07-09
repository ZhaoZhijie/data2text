import sys, codecs, json, os
from collections import Counter, defaultdict
from nltk import sent_tokenize, word_tokenize
import numpy as np
import h5py
# import re
import random
import math
from text2num import text2num, NumberException
import argparse

random.seed(2)


prons = set(["he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"]) # leave out "it"
singular_prons = set(["he", "He", "him", "Him", "his", "His"])
plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])
days_set = set(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

def get_ents(dat):
    players = set()
    teams = set()

    for thing in dat:
        teams.add(thing["vis_name"])
        teams.add(thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["vis_city"] + " " + thing["vis_name"])
        teams.add(thing["vis_city"] + " " + thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["home_name"])
        teams.add(thing["home_line"]["TEAM-NAME"])
        teams.add(thing["home_city"] + " " + thing["home_name"])
        teams.add(thing["home_city"] + " " + thing["home_line"]["TEAM-NAME"])
        # special case for this
        if thing["vis_city"] == "Los Angeles":
            teams.add("LA" + thing["vis_name"])
        if thing["home_city"] == "Los Angeles":
            teams.add("LA" + thing["home_name"])
        players.update(thing["box_score"]["PLAYER_NAME"].values())

    for entset in [players, teams]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        entset.add(piece)

    all_ents = players | teams

    return all_ents, players, teams

def get_cities(dat):
    cities = set()
    for thing in dat:
        # sometimes team_city is different
        cities.add(thing["home_city"])
        cities.add(thing["vis_city"])
        if "TEAM_CITY" in thing["box_score"].keys():
            cities.update(thing["box_score"]["TEAM_CITY"].values())
    return cities

def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None, since
    # we'll catch it anyway
    for j in range(len(curr_ents)-1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in range(len(prev_ents)-1, len(prev_ents)-1-max_back, -1):
            for j in range(len(prev_ents[i])-1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
        players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    ents_list = list(all_ents)
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(sent[i], players, teams, cities, sent_ents, prev_ents)
                if referent is None:
                    sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
                else:
                    #print "replacing", sent[i], "with", referent[2], "in", " ".join(sent)
                    sent_ents.append((i, i+1, referent[2], False)) # pretend it's not a pron and put in matching string
            else:
                sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
            i += 1
        elif sent[i] in all_ents: # findest longest spans; only works if we put in words...
            j = 1
            while i+j <= len(sent) and " ".join(sent[i:i+j]) in all_ents:
                # print("i:{} j:{} sent[i:i+j]={} ent_i:{} ent_i_j:{}".format(i,j,sent[i:i+j],ents_list[ents_list.index(sent[i])], ents_list[ents_list.index(" ".join(sent[i:i+j]))]))
                j += 1
            sent_ents.append((i, i+j-1, " ".join(sent[i:i+j-1]), False))
            i += j-1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = set(["three point", "three - point", "three - pt", "three pt"])
    return " ".join(sent[i:i+3]) not in ignores and " ".join(sent[i:i+2]) not in ignores


def extract_vals(sent, cities_set):
    sent_nums = []
    sent_days = []
    sent_cities = []
    i = 0
    ignores = set(["three point", "three-point", "three-pt", "three pt"])
    #print sent
    while i < len(sent):
        toke = sent[i]
        if toke in cities_set:
            sent_cities.append((i, i+1, toke))
            i += 1
        elif toke in days_set:
            sent_days.append((i, i+1, toke))
            i += 1
        else:
            a_number = False
            try:
                itoke = int(toke)
                a_number = True
            except ValueError:
                pass
            if a_number:
                sent_nums.append((i, i+1, int(toke)))
                i += 1
            elif toke in number_words and not annoying_number_word(sent, i): # get longest span  (this is kind of stupid)
                j = 1
                while i+j <= len(sent) and sent[i+j] in number_words and not annoying_number_word(sent, i+j):
                    j += 1
                try:
                    sent_nums.append((i, i+j, text2num(" ".join(sent[i:i+j]))))
                except NumberException:
                    print(sent)
                    print(sent[i:i+j])
                    assert False
                i += j
            else:
                i += 1
    return sent_nums, sent_days, sent_cities


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].items():
         if entname == v:
             keys.append(k)
            #  #判断全名是否都是由第一个第二个名字组合的
            #  FIRST_NAME = bs["FIRST_NAME"][k]
            #  SECOND_NAME = bs["SECOND_NAME"][k]
            #  if bs["PLAYER_NAME"][k] != FIRST_NAME + " " + SECOND_NAME:
            #      print("名字不是组合的 FIRST {} SECOND {} ALL{}".format(FIRST_NAME, SECOND_NAME, bs["PLAYER_NAME"][k]))

    if len(keys) == 0:
        for k,v in bs["SECOND_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1: # take the earliest one
            keys.sort(key = lambda x: int(x))
            keys = keys[:1]
            #print "picking", bs["PLAYER_NAME"][keys[0]]
    if len(keys) == 0:
        for k,v in bs["FIRST_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1: # if we matched on first name and there are a bunch just forget about it
            return None
    #if len(keys) == 0:
        #print "Couldn't find", entname, "in", bs["PLAYER_NAME"].values()
    assert len(keys) <= 1, entname + " : " + str(bs["PLAYER_NAME"].values())
    return keys[0] if len(keys) > 0 else None

def is_team_home(teamname, entry):
    is_home = None
    namepieces = teamname.split()
    if namepieces[-1] in entry["home_name"]:
        is_home = True
    elif namepieces[-1] in entry["vis_name"]:
        is_home = False
    elif "LA" in namepieces[0]:
        if entry["home_city"] == "Los Angeles":
            is_home = True
        elif entry["vis_city"] == "Los Angeles":
            is_home = False
    return is_home

def get_rels(entry, ents, nums, days, cities, players_set, teams_set, cities_set):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    teams = []
    entry_keys = entry.keys()
    for i, ent in enumerate(ents):
        if ent[3]: # pronoun
            continue # for now
        entname = ent[2]
        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players_set and entname not in cities_set and entname not in teams_set:
            pidx = get_player_idx(bs, entname)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if pidx is not None: # player might not actually be in the game or whatever
                    for colname, col in bs.items():
                        if col[pidx] == strnum: # allow multiple for now
                            rels.append((ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None))
        else: # has to be team
            teams.append(ent)
            entpieces = entname.split()
            linescore = None
            is_home = is_team_home(entname, entry)
            linescore = entry["home_line"] if is_home else entry["vis_line"]
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if linescore is not None:
                    for colname, val in linescore.items():
                        if val == strnum:
                            #rels.append((ent, numtup, "TEAM-" + colname, is_home))
                            # apparently I appended TEAM- at some pt...
                            rels.append((ent, numtup, colname, is_home))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None)) # should i specialize the NONE labels too?
            next_key = "home_next_game" if is_home else "vis_next_game"
            for j, daytup in enumerate(days):
                found = False
                if "game" in entry_keys and entry["game"]["DAYNAME"] == daytup[2] and is_home:
                    rels.append((ent, daytup, "DAYNAME", is_home))
                    found = True
                if next_key in entry_keys and entry[next_key]["NEXT-DAYNAME"] == daytup[2]:
                    rels.append((ent, daytup, "NEXT-DAYNAME", is_home))
                    found = True
                if not found:
                    rels.append((ent, daytup, "NONE", None))
            for j, citytup in enumerate(cities):
                found = False
                if "game" in entry_keys and entry["game"]["CITY"] == citytup[2] and is_home:
                    rels.append((ent, citytup, "CITY", is_home))
                    found = True
                if next_key in entry_keys and entry[next_key]["NEXT-CITY"] == citytup[2]:
                    rels.append((ent, citytup, "NEXT-CITY", is_home))
                    found = True
                if not found:
                    rels.append((ent, citytup, "NONE", None))
    #If a sentence contains two teams, we judge whether they are the opponents of the current game or the next game
    team_len = len(teams)
    if team_len >= 2:
        for i in range(team_len):
            for j in range(i+1, team_len):
                found = False
                team_name_i = teams[i][2]
                team_name_j = teams[j][2]
                is_i_home = is_team_home(team_name_i, entry)
                is_j_home = is_team_home(team_name_j, entry)
                i_next_key = "home_next_game" if is_i_home else "vis_next_game"
                j_next_key = "home_next_game" if is_j_home else "vis_next_game"
                if "game" in entry_keys and team_name_i == entry["game"]["HOME-TEAM"] and team_name_j == entry["game"]["VISITING-TEAM"]:
                    rels.append((teams[i], teams[j][0:3], "VISITING-TEAM", is_i_home))
                    found = True
                if "game" in entry_keys and team_name_j == entry["game"]["HOME-TEAM"] and team_name_i == entry["game"]["VISITING-TEAM"]:
                    rels.append((teams[j], teams[i][0:3], "VISITING-TEAM", is_j_home))
                    found = True 
                if i_next_key in entry_keys and team_name_i == entry[i_next_key]["NEXT-HOME-TEAM"] and team_name_j == entry[i_next_key]["NEXT-VISITING-TEAM"]:
                    rels.append((teams[i], teams[j][0:3], "NEXT-VISITING-TEAM", is_i_home))
                    found = True
                if j_next_key in entry_keys and team_name_j == entry[j_next_key]["NEXT-HOME-TEAM"] and team_name_i == entry[j_next_key]["NEXT-VISITING-TEAM"]:
                    rels.append((teams[j], teams[i][0:3], "NEXT-VISITING-TEAM", is_j_home))
                    found = True
                if not found:
                    rels.append((teams[i], teams[j][0:3], "NONE", None))
    return rels

def append_candidate_rels(entry, summ, all_ents, prons, players_set, teams_set, cities_set, candrels):
    """
    appends tuples of form (sentence_tokens, [rels]) to candrels
    """
    sents = sent_tokenize(summ)
    for j, sent in enumerate(sents):
        #tokes = word_tokenize(sent)
        tokes = sent.split()
        ents = extract_entities(tokes, all_ents, prons)
        nums, days, cities = extract_vals(tokes, cities_set)
        rels = get_rels(entry, ents, nums, days, cities, players_set, teams_set, cities_set)
        if len(rels) > 0:
            candrels.append((tokes, rels))
    return candrels

def get_datasets(path="./boxscore-data/rotowire"):
    print("get dataset start")
    with codecs.open(os.path.join(path, "train.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    all_ents, players_set, teams_set = get_ents(trdata)
    cities_set = get_cities(trdata)

    with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
        valdata = json.load(f)

    with codecs.open(os.path.join(path, "test.json"), "r", "utf-8") as f:
        testdata = json.load(f)
        
    extracted_stuff = []
    datasets = [trdata, valdata, testdata]
    for dataset in datasets:
        nugz = []
        for i, entry in enumerate(dataset):
            summ = " ".join(entry['summary'])
            append_candidate_rels(entry, summ, all_ents, prons, players_set, teams_set, cities_set, nugz)

        extracted_stuff.append(nugz)

    del all_ents
    del players_set
    del teams_set
    del cities_set
    return extracted_stuff

def append_to_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    for rel in tup[1]:
        ent, num, label, idthing = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in range(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in range(max_len)]
        numdists.append(num_dists)
        labels.append(labeldict[label])


def append_multilabeled_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    # get all the labels for the same rel
    unique_rels = defaultdict(list)
    for rel in tup[1]:
        ent, num, label, idthing = rel
        unique_rels[ent, num].append(label)

    for rel, label_list in unique_rels.items():
        ent, num = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in range(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in range(max_len)]
        numdists.append(num_dists)
        for label in label_list:
            if label not in labeldict.keys():
                print("label {}, labeldict {}".format(label, labeldict.keys()))
        labels.append([labeldict[label] for label in label_list])

def append_labelnums(labels):
    labelnums = [len(labellist) for labellist in labels]
    max_num_labels = max(labelnums)
    print("max num labels", max_num_labels)

    # append number of labels to labels
    for i, labellist in enumerate(labels):
        labellist.extend([-1]*(max_num_labels - len(labellist)))
        labellist.append(labelnums[i])

# for full sentence IE training
def save_full_sent_data(outfile, path="../boxscore-data/rotowire", multilabel_train=False, nonedenom=0):
    datasets = get_datasets(path)
    # make vocab and get labels
    word_counter = Counter()
    #统计训练数据集summary中所有出现的单词
    [word_counter.update(tup[0]) for tup in datasets[0]]
    workkeys = [key for key in word_counter.keys()]
    #将数量少于2的单词删掉
    for k in workkeys:
        if word_counter[k] < 2:
            del word_counter[k] # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i) for i, wrd in enumerate(word_counter.keys())))
    labelset = set()
    [labelset.update([rel[2] for rel in tup[1]]) for tup in datasets[0]]
    labeldict = dict(((label, i) for i, label in enumerate(labelset)))

    # save stuff
    trsents, trlens, trentdists, trnumdists, trlabels = [], [], [], [], []
    valsents, vallens, valentdists, valnumdists, vallabels = [], [], [], [], []
    testsents, testlens, testentdists, testnumdists, testlabels = [], [], [], [], []

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    print("max tr sentence length:", max_trlen)

    # do training data
    for tup in datasets[0]:
        if multilabel_train:
            append_multilabeled_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)
        else:
            append_to_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)

    if multilabel_train:
        append_labelnums(trlabels)
    print("nonedenom",nonedenom)
    if nonedenom > 0:
        # don't keep all the NONE labeled things
        none_idxs = [i for i, labellist in enumerate(trlabels) if labellist[0] == labeldict["NONE"]]
        random.shuffle(none_idxs)
        # allow at most 1/(nonedenom+1) of NONE-labeled
        num_to_keep = int(math.floor(float(len(trlabels)-len(none_idxs))/nonedenom))
        print("originally", len(trlabels), "training examples")
        print("keeping", num_to_keep, "NONE-labeled examples")
        ignore_idxs = set(none_idxs[num_to_keep:])

        # get rid of most of the NONE-labeled examples
        trsents = [thing for i,thing in enumerate(trsents) if i not in ignore_idxs]
        trlens = [thing for i,thing in enumerate(trlens) if i not in ignore_idxs]
        trentdists = [thing for i,thing in enumerate(trentdists) if i not in ignore_idxs]
        trnumdists = [thing for i,thing in enumerate(trnumdists) if i not in ignore_idxs]
        trlabels = [thing for i,thing in enumerate(trlabels) if i not in ignore_idxs]

    print(len(trsents), "training examples")

    # do val, which we also consider multilabel
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    for tup in datasets[1]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        append_multilabeled_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_vallen)

    append_labelnums(vallabels)

    print(len(valsents), "validation examples")

    # do test, which we also consider multilabel
    max_testlen = max((len(tup[0]) for tup in datasets[2]))
    for tup in datasets[2]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        append_multilabeled_data(tup, testsents, testlens, testentdists, testnumdists, testlabels, vocab, labeldict, max_testlen)

    append_labelnums(testlabels)

    print(len(testsents), "test examples")

    h5fi = h5py.File(outfile, "w")
    h5fi["trsents"] = np.array(trsents, dtype=int)
    h5fi["trlens"] = np.array(trlens, dtype=int)
    h5fi["trentdists"] = np.array(trentdists, dtype=int)
    h5fi["trnumdists"] = np.array(trnumdists, dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)

    h5fi["valsents"] = np.array(valsents, dtype=int)
    h5fi["vallens"] = np.array(vallens, dtype=int)
    h5fi["valentdists"] = np.array(valentdists, dtype=int)
    h5fi["valnumdists"] = np.array(valnumdists, dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    #h5fi.close()

    #h5fi = h5py.File("test-" + outfile, "w")
    h5fi["testsents"] = np.array(testsents, dtype=int)
    h5fi["testlens"] = np.array(testlens, dtype=int)
    h5fi["testentdists"] = np.array(testentdists, dtype=int)
    h5fi["testnumdists"] = np.array(testnumdists, dtype=int)
    h5fi["testlabels"] = np.array(testlabels, dtype=int)
    h5fi.close()
    ## h5fi["vallabelnums"] = np.array(vallabelnums, dtype=int)
    ## h5fi.close()

    # write dicts
    revvocab = dict(((v,k) for k,v in vocab.items()))
    revlabels = dict(((v,k) for k,v in labeldict.items()))
    with codecs.open(outfile.split('.')[0] + ".dict", "w+", "utf-8") as f:
        for i in range(0, len(revvocab)):
            f.write("%s %d \n" % (revvocab[i], i))

    with codecs.open(outfile.split('.')[0] + ".labels", "w+", "utf-8") as f:
        for i in range(0, len(revlabels)):
            f.write("%s %d \n" % (revlabels[i], i))


def prep_generated_data(genfile, dict_pfx, outfile, path="../boxscore-data/rotowire", test=False):
    # recreate vocab and labeldict
    vocab = {}
    with codecs.open(dict_pfx+".dict", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            vocab[pieces[0]] = int(pieces[1])

    labeldict = {}
    with codecs.open(dict_pfx+".labels", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            labeldict[pieces[0]] = int(pieces[1])
    with codecs.open(genfile, "r", "utf-8") as f:
        gens = f.readlines()
        if gens[-1] == "":
            gens.pop(-1)

    with codecs.open(os.path.join(path, "train.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    all_ents, players_set, teams_set = get_ents(trdata)
    cities_set = get_cities(trdata)

    valfi = "test.json" if test else "valid.json"
    with codecs.open(os.path.join(path, valfi), "r", "utf-8") as f:
        valdata = json.load(f)

    print("valdata len {}, gens len {}".format(len(valdata), len(gens)))
    assert len(valdata) == len(gens)

    nugz = [] # to hold (sentence_tokens, [rels]) tuples
    sent_reset_indices = {0:1} # sentence indices where a box/story is reset
    for i, entry in enumerate(valdata):
        summ = gens[i]
        append_candidate_rels(entry, summ, all_ents, prons, players_set, teams_set, cities_set, nugz)
        nugz_len = len(nugz)
        if nugz_len not in sent_reset_indices.keys():
            sent_reset_indices[nugz_len] = 0
        sent_reset_indices[nugz_len] += 1
        #注意有些样本的输出中可能没有提取出relation，那样nugz长度不会增长，这会导致后续评估时判断错误评估输入数据中包含的样本数量，比如旧数据集验证集366号样本
        # sent_reset_indices.add(len(nugz))
    # save stuff
    max_len = max((len(tup[0]) for tup in nugz))
    psents, plens, pentdists, pnumdists, plabels = [], [], [], [], []
    rel_reset_indices = []
    for t, tup in enumerate(nugz):
        if t in sent_reset_indices.keys(): # then last rel is the last of its box
            assert len(psents) == len(plabels)
            for k in range(sent_reset_indices[t]):
                rel_reset_indices.append(len(psents))
            del sent_reset_indices[t]
        append_multilabeled_data(tup, psents, plens, pentdists, pnumdists, plabels, vocab, labeldict, max_len)
    #有可能最后几个没有提取出关系，它们的终止长度是一样的，这种特殊情况之前的代码也没考虑到
    lastkeys = sent_reset_indices.keys()
    print("nugz length", len(nugz), "last key", lastkeys)
    assert len(lastkeys) == 1
    for i in range(sent_reset_indices.popitem()[1] - 1):
        rel_reset_indices.append(len(psents))
    print("rel_reset_indices length", len(rel_reset_indices))

    append_labelnums(plabels)

    print(len(psents), "prediction examples")

    h5fi = h5py.File(outfile, "w")
    h5fi["valsents"] = np.array(psents, dtype=int)
    h5fi["vallens"] = np.array(plens, dtype=int)
    h5fi["valentdists"] = np.array(pentdists, dtype=int)
    h5fi["valnumdists"] = np.array(pnumdists, dtype=int)
    h5fi["vallabels"] = np.array(plabels, dtype=int)
    h5fi["boxrestartidxs"] = np.array(np.array(rel_reset_indices)+1, dtype=int) # 1-indexed
    h5fi.close()

parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-input_path', type=str, default="",
                    help="path to input")
parser.add_argument('-output_fi', type=str, default="",
                    help="desired path to output file")
parser.add_argument('-gen_fi', type=str, default="",
                    help="path to file containing generated summaries")
parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                    help="prefix of .dict and .labels files")
parser.add_argument('-mode', type=str, default='ptrs',
                    choices=['make_ie_data', 'prep_gen_data'],
                    help="what utility function to run")
parser.add_argument('-test', action='store_true', help='use test data')

args = parser.parse_args()

if args.mode == 'make_ie_data':
    save_full_sent_data(args.output_fi, path=args.input_path, multilabel_train=True)
elif args.mode == 'prep_gen_data':
    prep_generated_data(args.gen_fi, args.dict_pfx, args.output_fi, path=args.input_path,
                        test=args.test)
