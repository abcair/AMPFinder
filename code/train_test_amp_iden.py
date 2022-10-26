

import numpy as np
from Bio import SeqIO
from propy import PyPro
import blosum as bl
from protlearn import  features
import joblib

std=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def read_data(path):
    res = {}
    rx = list(SeqIO.parse(path,format="fasta"))
    for x in rx:
        id = str(x.id)
        seq = str(x.seq).upper()
        seq = "".join([x for x in list(seq) if x in std])
        res[id]=seq
    return res

def get_ctd(seq):
    DesObject = PyPro.GetProDes(seq)
    res = list(DesObject.GetCTD().values())
    return res

def get_DPC(seq):
    res = list(PyPro.GetProDes(seq).GetDPComp().values())
    return res

def get_aaindex1(seq):
    res = list(features.aaindex1(seq)[0].flatten())
    return res

def get_blosum_80(seq):
    mat = bl.BLOSUM(80)
    res = np.zeros((20,20))
    i = 0
    while i+1 <len(seq):
        x = seq[i]
        y = seq[i+1]
        val = mat[x+y]
        k = std.index(x)
        m = std.index(y)
        res[k,m] = val
        i = i+1
    res = res.flatten().tolist()
    return res

def encoding(seq):
    res = get_ctd(seq) + get_DPC(seq) + get_aaindex1(seq)  + get_blosum_80(seq)
    return res

from multiprocessing import Pool,cpu_count
if __name__ == '__main__':

    avp_path = "../data/D1/3594-Samp.fasta"
    nonavp_path = "../data/D1/3925-Snonamp.fasta"

    avp_seqs    = list(read_data(avp_path).values())
    nonavp_seqs = list(read_data(nonavp_path).values())


    pool = Pool(cpu_count())
    avp_data    = np.array(pool.map(encoding, avp_seqs))
    nonavp_data = np.array(pool.map(encoding, nonavp_seqs))
    data = np.concatenate([avp_data,nonavp_data])
    label = np.array([1]*len(avp_data)+[0]*len(nonavp_data))

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=12000,n_jobs=-1)
    clf.fit(X=data,
              y=label)

    joblib.dump(clf, 'AMPFinder.identify.rf')




