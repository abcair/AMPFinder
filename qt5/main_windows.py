import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from Bio import SeqIO
from qt_material import apply_stylesheet
import numpy as np
import blosum as bl
from propy import PyPro
from protlearn import features
import joblib

from transformers import T5Tokenizer, T5Model
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM,FeatureExtractionPipeline

import torch
torch.set_num_threads(10)

import tensorflow.keras as keras

app = QApplication([])
ui = QUiLoader().load('./AMPFinder.ui')

rf = joblib.load("./model/AMPFinder.identify.rf")
deep_model = keras.models.load_model("./model/AMPFinder.fun.deep.model")

#pip install sentencepiece
tokenizer_T5_bfd = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
model_T5_bfd = T5Model.from_pretrained("Rostlab/prot_t5_xl_bfd")

tokenizer_T5_uniref50 = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
model_T5_uniref50 = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50")

tokenizer_onto = AutoTokenizer.from_pretrained("zjukg/OntoProtein")
model_onto = AutoModelForMaskedLM.from_pretrained("zjukg/OntoProtein")
bert_model_onto = FeatureExtractionPipeline(model=model_onto, tokenizer=tokenizer_onto)

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

def get_t5xl_bfd_bert(seq):
    sequences_Example = [" ".join(list(seq))]
    ids = tokenizer_T5_bfd.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])
    with torch.no_grad():
        embedding = model_T5_bfd(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy()
    decoder_embedding = embedding[0].cpu().numpy()
    res = encoder_embedding
    res = np.array(res)[0,1:,:]
    xx = np.zeros((100,1024))
    tmp = res[0:100,:]
    xx[:tmp.shape[0],:]=tmp
    return xx

def get_t5xl_uniref50_bert(seq):
    sequences_Example = [" ".join(list(seq))]
    ids = tokenizer_T5_uniref50.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])
    with torch.no_grad():
        embedding = model_T5_uniref50(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy()
    decoder_embedding = embedding[0].cpu().numpy()

    res = encoder_embedding
    res = np.array(res)[0,1:,:]
    xx = np.zeros((100,1024))
    tmp = res[0:100,:]
    xx[:tmp.shape[0],:]=tmp
    return xx

def get_onto_bert(seq):
    seq = " ".join(list(seq))
    res = bert_model_onto(seq)
    res = np.array(res)[0,:,:]
    xx = np.zeros((100,30))
    tmp = res[0:100,:]
    xx[:tmp.shape[0],:] = tmp
    return xx

def encoding_bert(seq):
    res1 = get_onto_bert(seq)
    print(res1.shape)
    res2 = get_t5xl_bfd_bert(seq)
    print(res2.shape)
    res3 = get_t5xl_uniref50_bert(seq)
    print(res3.shape)
    res = np.concatenate([res1,res2,res3],axis=-1)
    res = res[np.newaxis,:,:]
    return res

def pred_encoding(seq):
    std = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    def get_blosum_80(seq):
        mat = bl.BLOSUM(80)
        res = np.zeros((20, 20))
        i = 0
        while i + 1 < len(seq):
            x = seq[i]
            y = seq[i + 1]
            val = mat[x + y]
            k = std.index(x)
            m = std.index(y)
            res[k, m] = val
            i = i + 1
        res = res.flatten().tolist()
        return res


    def get_ctd(seq):
        DesObject = PyPro.GetProDes(seq)  # construct a GetProDes object
        res = list(DesObject.GetCTD().values())  # calculate 147 CTD descriptors
        return res

    def get_DPC(seq):
        # Dipeptide composition descriptors (400)
        res = list(PyPro.GetProDes(seq).GetDPComp().values())
        return res

    def get_aaindex1(seq):
        res = list(features.aaindex1(seq)[0].flatten())
        return res

    def encoding(seq):
        res = get_ctd(seq) + get_DPC(seq) + get_aaindex1(seq) + get_blosum_80(seq)
        return res
    vec = np.array(encoding(seq))
    return vec

def predict_amp(info,tt):
    open("tmp.fa",'w').write(info)
    rx = SeqIO.parse("./tmp.fa",format="fasta")
    tt = float(tt)
    ress= []
    for x in rx:
        id  = str(x.id)
        seq = str(x.seq)
        vec = pred_encoding(seq)
        res = rf.predict_proba(vec.reshape(1,-1))[:,1][0]
        label = "1,AMP" if res>tt else "0,not AMP"
        ress.append(",".join([id,seq,str(res),label])+"\n")
    return ress

def show_identify():
    info= ui.inp_fa_pred.toPlainText()
    tt = ui.PT.text()
    if info and tt:
        ress = predict_amp(info,tt)
        ressw = "".join(ress)
        ui.AMPRes.setText(ressw)

def predict_fun(info,tt):
    open("tmp.fa",'w').write(info)
    rx = SeqIO.parse("./tmp.fa",format="fasta")
    tt = float(tt)
    ress= []
    for x in rx:
        id  = str(x.id)
        seq = str(x.seq)
        vec = encoding_bert(seq)
        res = deep_model.predict(vec)[0]
        print(res)

        fx = ["bacterial",
        "viral",
        "parasital",
        "HIV",
        "cancer",
        "MRSA",
        "fungal",
        "endotoxin",
        "biofilm",
        "Chemotactic"]

        tmp = []
        for i in range(len(res)):
            ss = " ".join([fx[i],str(res[i]),"1" if res[i]>tt else "0"])
            tmp.append(ss)
        ww = ",".join(tmp)
        ress.append(id+","+seq+","+ww+"\n")
    return ress

def show_predict_fun():
    info= ui.inp_fa_pred_fun.toPlainText()
    tt = ui.FT.text()
    if info and tt:
        ress = predict_fun(info,tt)
        ressw = "".join(ress)
        ui.AMPFunRes.setText(ressw)

styles=['dark_amber.xml',
 'dark_blue.xml',
 'dark_cyan.xml',
 'dark_lightgreen.xml',
 'dark_pink.xml',
 'dark_purple.xml',
 'dark_red.xml',
 'dark_teal.xml',
 'dark_yellow.xml',
 'light_amber.xml',
 'light_blue.xml',
 'light_cyan.xml',
 'light_cyan_500.xml',
 'light_lightgreen.xml',
 'light_pink.xml',
 'light_purple.xml',
 'light_red.xml',
 'light_teal.xml',
 'light_yellow.xml']

apply_stylesheet(app, theme='light_pink.xml')

ui.but_pred.clicked.connect(show_identify)
ui.but_fun.clicked.connect(show_predict_fun)
ui.show()
app.exec_()