import nltk
from nltk.tokenize import word_tokenize
import os
import re
import numpy as np
import json
import pandas as pd
import csv

#nltk.download('punkt')

file_name="train"

f = open(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+"2.txt"), 'r')
corpus= json.loads(f.read())
f.close()

tokenized_sent = []
for c in corpus:
    tokenized_sent.append(word_tokenize(c.lower()))
tokenized_sent

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
#print(tagged_data[0:10])

# ## Train doc2vec model
#model = Doc2Vec(tagged_data, vector_size = 100, window = 2, min_count = 1, epochs = 100)



# '''
# vector_size = Dimensionality of the feature vectors.
# window = The maximum distance between the current and predicted word within a sentence.
# min_count = Ignores all words with total frequency lower than this.
# alpha = The initial learning rate.
# '''

#model.save(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+"2.model"))

model = Doc2Vec.load(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+"2.model"))
#model.build_vocab(tagged_data, update=True)
#model.train(tagged_data,total_examples=1, epochs=1)

#print(model)


# w1 = word_tokenize("So EJ Manuel is the 1st QB take".lower())
# w2 = word_tokenize("EJ Manuel is the 1st qb taken huh".lower())


# print(model.similarity_unseen_docs(w1,w2 ))




df_dev = pd.read_csv("data_processed\\dev.csv",names=['sentence1', 'sentence2', 'label'])

sentences1_dev = df_dev['sentence1'].values
sentences2_dev = df_dev['sentence2'].values
y_dev = df_dev['label'].values

#print(type(y_dev))

# resultlist=[]
# resultlist.append(["result"])

# for s in range(len(sentences1_dev)):
#     w1 = word_tokenize(sentences1_dev[s].lower())
#     w2 = word_tokenize(sentences2_dev[s].lower())

#     result= model.similarity_unseen_docs(w1,w2 )
#     if result > .35:
#         resultlist.append([1])
#     else:
#         resultlist.append([0])
        
# with open(os.path.join(os.path.dirname(__file__), "data_processed\\result.csv"), 'a', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(resultlist) 
    
    
 
df_result = pd.read_csv("data_processed\\result.csv",names=['result'])  

y_result = df_result['result'].values
#print(y_result)


# from sklearn.metrics import f1_score, precision_score, recall_score

# print(y_dev)

# print(set(y_dev) - set(y_result))

# print(f1_score(y_dev, y_result, average="macro"))
# print(precision_score(y_dev, y_result, average="macro"))
# print(recall_score(y_dev,y_result, average="macro"))   



tp = 0
tn = 0
fp = 0
fn = 0

for ts,ps in zip(y_dev[1:],  y_result[1:]):
    t=int(ts)
    p=int(ps)
    if t == p:
        if p == 1:
            tp += 1
        else:
            tn += 1
    else:
        if p == 1:
            fn += 1
        else:
            fp += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1= (2 * precision * recall) / (precision + recall)
print("Accuracy: "+ str(accuracy))
print("Precision: "+ str(precision))
print("Recall: "+ str(recall))
print("F1: "+ str(f1))


# true_values = np.array([y_dev])
# predictions = np.array([y_result])


# N = true_values.shape[1]
# accuracy = (true_values == predictions).sum() / N
# TP = ((predictions == 1) & (true_values == 1)).sum()
# FP = ((predictions == 1) & (true_values == 0)).sum()
# precision = TP / (TP+FP)

# print(precision)