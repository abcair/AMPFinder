from sklearn.model_selection import train_test_split
from Bio import SeqIO
import numpy as np
import tensorflow.keras as keras

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

label_path = "../../data/D2/new_fun.label"
rx  = open(label_path).readlines()

labels = []
for x in rx:
    x = x.strip()
    tmp = [float(xx) for xx in  x.split(",")[1:]]
    labels.append(tmp)
labels = np.array(labels)

onto_bert = np.load("../../data/D2/OntoProtein.npy")
xl_bfd_bert = np.load("../../data/D2/xl_bfd.npy")
xl_uniref50_bert = np.load("../../data/D2/xl_uniref50.npy")
bert_data = np.concatenate([onto_bert, xl_bfd_bert, xl_uniref50_bert], axis=-1)

label = labels

inx2 = keras.layers.Input(shape=(100, 2078))

x1 = inx2
x1 = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=False))(x1)

x2 = inx2
x2 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=False))(x2)

x3 = inx2
x3 = keras.layers.Dense(units=256, activation="relu")(x3)
x31 = keras.layers.GlobalMaxPool1D()(x3)
x32 = keras.layers.GlobalAveragePooling1D()(x3)
x3 = keras.layers.Add()([x31, x32])

x = keras.layers.Add()([x1, x2, x3])
xs = keras.activations.softmax(x)
x = keras.layers.Concatenate()([x1, x2, x3, xs])
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(units=256, activation="relu")(x)
x = keras.layers.Dense(units=64, activation="relu")(x)
outx = keras.layers.Dense(units=10, activation="sigmoid")(x)
model = keras.Model(inputs=[inx2], outputs=[outx])
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss=keras.losses.binary_crossentropy,
              metrics=[keras.metrics.BinaryAccuracy()], )

estop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model.fit(x=[bert_data],
          y=label,
          validation_split=0.1,
          shuffle=True,
          epochs=10000,
          batch_size=64,
          callbacks=[estop],
          )

model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),
              loss=keras.losses.binary_crossentropy,
              metrics=[keras.metrics.BinaryAccuracy()],)

estop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model.save("AMPFinder.fun.deep.model")




