import os
from Bio import SeqIO
import pickle
import numpy as np
import argparse
import pdb

def hot_encode(sequence):
    seq_encoded = np.zeros((len(sequence), 4))
    dict_nuc = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    i = 0
    for l in sequence:
        if(l.upper() in dict_nuc.keys()):
            seq_encoded[i][dict_nuc[l.upper()]] = 1
            i = i+1
        else:
            return []
    return seq_encoded

def CKSNAP(sequence, k):
    CKSNAP_encoded = []
    for i in range(1, k + 1):
        dic = {'AA': 0, "AC": 0, "AG": 0, "AT": 0, "CA": 0, "CC": 0, "CG": 0, "CT": 0,
               "GA": 0, "GC": 0, "GG": 0, "GT": 0, "TA": 0, "TC": 0, "TG": 0, "TT": 0}
        for j in range(len(sequence)):
            dic[sequence[j].upper() + sequence[j + i].upper()] += 1
        knp = []
        for v in dic.values():
            v /= len(sequence) - k - 1
            knp.append(v)
        CKSNAP_encoded.append(knp)
    return CKSNAP_encoded

def NCP(sequence):
    NCP_encoded = []
    dic = {
        "A": [1, 1, 1],
        "C": [0, 1, 0],
        "G": [1, 0, 0],
        "T": [0, 0, 1]
    }
    for l in sequence:
        if(l.upper() in dic.keys()):
            NCP_encoded.append(dic[l.upper()])
        else:
            return []
    return NCP_encoded

def EIIP(sequence):
    EIIP_encoded = []
    dic = {
        "A": 0.1260,
        "C": 0.1340,
        "G": 0.0806,
        "T": 0.1335
    }
    for l in sequence:
        if l.upper() in dic.keys():
            EIIP_encoded.append([dic[l.upper()]])
        else:
            return []
    return EIIP_encoded

def ANF(sequence):
    ANF_encoded = []
    for i in range(len(sequence)):
        ANF_score = sequence[: i + 1].count(sequence[i])
        ANF_encoded.append([ANF_score])
    return ANF_encoded

def ENAC(sequence, w):
    ENAC_encoded = []
    for i in range(len(sequence)):
        f = []
        f.append(sequence[i: i + w].count("A"))
        f.append(sequence[i: i + w].count("C"))
        f.append(sequence[i: i + w].count("G"))
        f.append(sequence[i: i + w].count("T"))
        ENAC_encoded.append(f)
    return ENAC_encoded

def PseTNC(sequence):
    dic = dict()
    PseTNC_encoded = []
    lst = ['A', 'C', 'G', 'T']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dic[lst[i] + lst[j] + lst[k]] = 0
    for i in range(len(sequence) - 2):
        s = sequence[i].upper() + sequence[i + 1].upper() + sequence[i + 2].upper()
        dic[s] += 1
    data = dict(sorted(dic.items(), key=lambda x: x[0]))
    for k in data.values():
        k /= 145
        PseTNC_encoded.append([k])
    return PseTNC_encoded

parser = argparse.ArgumentParser(description='Generating Pickle encoded File from Fasta')
parser.add_argument('-p', '--path', dest='path', type=str, default=r"D:\data\setting1",
                    help='Fasta File Path')
parser.add_argument('-f', '-fas', dest='fasName', type=str, default="nucleosomes_vs_linkers_elegans.fas",
                    help='Fasta filename')
parser.add_argument('-o',  '--out', dest='outDir', type=str, default="D:\data\setting1/pickle_H",
                    help='Output file Path')
parser.add_argument('-n', '--nuc', dest='nucPickle', type=str, default="nuc.pickle",
                    help='Output Nucleosome filename')
parser.add_argument('-l', '--lin', dest='linkPickle', type=str, default="link.pickle",
                    help='Output Linker filename')

args = parser.parse_args()
inPath = args.path
outPath = args.outDir
fasName = args.fasName
nucPickle = args.nucPickle
linkPickle = args.linkPickle

del args

if(not os.path.isdir(outPath)):
    os.mkdir(outPath)

nucList = []
linkList = []
nuc_one_hot_List = []
link_one_hot_List = []
nuc_ANF_List = []
link_ANF_List = []
nuc_NCP_List = []
link_NCP_List = []
nuc_EIIP_List = []
link_EIIP_List = []
nuc_PseTNC_List = []
link_PseTNC_List = []
nuc_ENAC_List = []
link_ENAC_List = []
nuc_Species_List = []
link_Species_List = []


fastaSequences = SeqIO.parse(open(os.path.join(inPath, fasName)), 'fasta')
for fasta in fastaSequences:
    name, sequence = fasta.id, str(fasta.seq)
    if "nucleosomal" in name:
        nucList.append(sequence)
        nuc_one_hot_List.append(hot_encode(sequence))
        nuc_ANF_List.append(ANF(sequence))
        nuc_NCP_List.append(NCP(sequence))
        nuc_EIIP_List.append(EIIP(sequence))
        nuc_PseTNC_List.append(PseTNC(sequence))
        nuc_ENAC_List.append(ENAC(sequence, 2))

    else:
        linkList.append(sequence)
        link_one_hot_List.append(hot_encode(sequence))
        link_ANF_List.append(ANF(sequence))
        link_NCP_List.append(NCP(sequence))
        link_EIIP_List.append(EIIP(sequence))
        link_PseTNC_List.append(PseTNC(sequence))
        link_ENAC_List.append(ENAC(sequence, 2))


print(len(nuc_ANF_List))
print(len(link_ANF_List))
print(len(nuc_Species_List))
print(len(link_Species_List))

pdb.set_trace()
with open(os.path.join(outPath, nucPickle), "wb") as fp:
    pickle.dump(nucList, fp)
with open(os.path.join(outPath, linkPickle), "wb") as fp:
    pickle.dump(linkList, fp)

with open(os.path.join(outPath, "one_hot_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_one_hot_List, fp)
with open(os.path.join(outPath, "one_hot_"+linkPickle), "wb") as fp:
    pickle.dump(link_one_hot_List, fp)

with open(os.path.join(outPath, "ANF_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_ANF_List, fp)
with open(os.path.join(outPath, "ANF_"+linkPickle), "wb") as fp:
    pickle.dump(link_ANF_List, fp)

with open(os.path.join(outPath, "NCP_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_NCP_List, fp)
with open(os.path.join(outPath, "NCP_"+linkPickle), "wb") as fp:
    pickle.dump(link_NCP_List, fp)

with open(os.path.join(outPath, "EIIP_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_EIIP_List, fp)
with open(os.path.join(outPath, "EIIP_"+linkPickle), "wb") as fp:
    pickle.dump(link_EIIP_List, fp)

with open(os.path.join(outPath, "PseTNC_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_PseTNC_List, fp)
with open(os.path.join(outPath, "PseTNC_"+linkPickle), "wb") as fp:
    pickle.dump(link_PseTNC_List, fp)

with open(os.path.join(outPath, "ENAC_"+nucPickle), "wb") as fp:
    pickle.dump(nuc_ENAC_List, fp)
with open(os.path.join(outPath, "ENAC_"+linkPickle), "wb") as fp:
    pickle.dump(link_ENAC_List, fp)

