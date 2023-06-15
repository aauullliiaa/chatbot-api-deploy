import json
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open(r'./intents.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

dic = {"tag": [], "patterns": [], "responses": [], "next-patterns": []}
for i in range(len(df)):
    nxt_ptrns = df[df.index == i]['next-patterns'].values[0]
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        dic['next-patterns'].append(nxt_ptrns)

df = pd.DataFrame.from_dict(dic)

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()

vacab_size = len(tokenizer.word_index)
ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])