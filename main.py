import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import joblib

spam_file_path = "emails.csv"
df = pd.read_csv(spam_file_path)
x, y = df["text"], df["spam"]

#Split the dataframe to two parts(train = 80%, test=20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=69)

#Word Embeddings/NLP 
vec = TfidfVectorizer()
x_train_num = vec.fit_transform(x_train)
x_test_num = vec.transform(x_test)

#---Train---

#Naive Bayes
model1 = MultinomialNB()
model1.fit(x_train_num, y_train)

#Logistic Regression
model2 = LogisticRegression()
model2.fit(x_train_num, y_train)

#LinearSVC
model3 = LinearSVC(class_weight="balanced")
model3.fit(x_train_num, y_train)

#Predict
preds1 = model1.predict(x_test_num)
preds2 = model2.predict(x_test_num)
preds3 = model3.predict(x_test_num)

#Print result from 3 models

#NB
print(classification_report(y_test, preds1))
print(confusion_matrix(y_test, preds1))
wrong1 = x_test[preds1 != y_test]
for text, true_label in zip(wrong1, y_test[preds1 != y_test]):
    print(true_label, "|", text[:300])
    print("----")

#LR
print(classification_report(y_test, preds2))
print(confusion_matrix(y_test, preds2))
wrong2 = x_test[preds2 != y_test]
for text, true_label in zip(wrong2, y_test[preds2 != y_test]):
    print(true_label, "|", text[:300])
    print("----")

#LinearSVC
print(classification_report(y_test, preds3))
print(confusion_matrix(y_test, preds3))
wrong3 = x_test[preds3 != y_test]
for text, true_label in zip(wrong3, y_test[preds3 != y_test]):
    print(true_label, "|", text[:300])
    print("----")


joblib.dump(vec, "vec.pkl")
joblib.dump(model1, "nb.pkl")
joblib.dump(model2, "lr.pkl")
joblib.dump(model3, "svc.pkl")

vec = joblib.load("vec.pkl")
inbox = pd.read_csv("small_inbox.csv")
inbox_num = vec.transform(inbox["text"])

for name in ["nb", "lr", "svc"]:
    model = joblib.load(f"{name}.pkl")
    preds = model.predict(inbox_num)
    print(f"{name}: {preds.sum()} of {len(preds)} flagged as spam")