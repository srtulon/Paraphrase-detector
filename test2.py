import pandas as pd
import os



df_list = []

df = pd.read_csv("data_processed\\train.csv",names=['sentence1', 'sentence2', 'label'])
df_list.append(df)

df = pd.concat(df_list)
#print(df.iloc[0])


# sentences = ['John likes ice cream', 'John hates chocolate.']

# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(min_df=0, lowercase=False)

# vectorizer.fit(sentences)
# vectorizer.transform(sentences).toarray()

# print(vectorizer.vocabulary_)



sentences1_train = df['sentence1'].values
sentences2_train = df['sentence2'].values
y_train = df['label'].values

#print("hi")

df_list = []

df_dev = pd.read_csv("data_processed\\dev.csv",names=['sentence1', 'sentence2', 'label'])

#df = pd.concat(df_list)
#print(df.iloc[0])

sentences1_dev = df_dev['sentence1'].values
sentences2_dev = df_dev['sentence2'].values
y_dev = df_dev['label'].values

from sklearn.feature_extraction.text import CountVectorizer

# FEATURES_ARR = ['feature1', 'feature2', 'feature3']
# lrModel = LinearRegression(normalize=True)
# lrModel.fit(X_train[FEATURES_ARR], Y_train)

vectorizer = CountVectorizer()
vectorizer.fit(sentences1_train)
vectorizer.fit(sentences2_train)


vectorizer.fit(sentences1_dev)
vectorizer.fit(sentences2_dev)


X_train1 = vectorizer.transform(sentences1_train)
X_train2 = vectorizer.transform(sentences2_train)

#print(sentences1_dev)
# print("1*"*70)
# print(X_train1[100])
# print("2*"*70)
# print(X_train2[100])
# print("#"*70)



X_train = vectorizer.transform(sentences1_train+sentences2_train)
#print("hi2",X_train[100])




X_dev  = vectorizer.transform(sentences1_dev+sentences2_dev)

X_train_arr = [vectorizer.transform(sentences1_train), vectorizer.transform(sentences2_train)]

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 10), random_state=1,max_iter=100)

# from sklearn.linear_model import LogisticRegression

# classifier = LogisticRegression()


classifier.fit(X_train, y_train)

score = classifier.score(X_dev, y_dev)

#print("Accuracy:", score)

# from tensorflow import keras
# import numpy as np


# merged_array = np.stack([vectorizer.transform(sentences1_train),vectorizer.transform(sentences2_train)], axis=0)

# model0 = keras.Sequential([
# keras.layers.Dense(2, input_dim=2, activation=keras.activations.sigmoid, use_bias=True),
# keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True),
# ])

# model0.compile(
#     optimizer = keras.optimizers.RMSprop(lr=0.02,rho=0.9,epsilon=None,decay=0),
#     loss = 'mean_squared_error',
#     metrics=['accuracy']
# )

#model0.fit(np.asarray(merged_array).astype(np.float32),y_train, batch_size=16, epochs=100)

# from sklearn.metrics import f1_score, precision_score, recall_score

y_pred = classifier.predict(X_dev)

#print (X_dev)

# print(f1_score(y_dev, y_pred, average="micro"))
# print(precision_score(y_dev, y_pred, average="micro"))
# print(recall_score(y_dev, y_pred, average="micro"))   

tp = 0
tn = 0
fp = 0
fn = 0

for ts,ps in zip(y_dev[1:],  y_pred[1:]):
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
