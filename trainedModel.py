import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pickle

"""Reading the initial dataset"""
df_initial=pd.read_csv("language_dataset.csv")

"""Modifying the dataset to make it easier to classify"""
lang_dict = {}
for lang in df_initial.columns:
  lang_list = []
  for row in df_initial[lang]:
    lang_dict[row] = lang
df_dict = {'language':lang_dict.values(),'text':lang_dict.keys()}
df = pd.DataFrame.from_dict(df_dict)

"""Using CountVectorizer to convert the phrases to vectors so that the model can process it"""
x = np.array(df['text'])
y = np.array(df['language'])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=41)

"""Training a Multinomial Naive Bayes model"""
model = MultinomialNB(alpha=1)
model.fit(X_train,y_train)

"""Pickling the trained model"""
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

"""Pickling the vectorizer"""
with open('vectorizer.pkl','wb') as f:
   pickle.dump(cv,f)

"""Taking user input of a sentence and using the fitted model to predict the language of the sentence"""
""" user_input = input("Enter a sentence: ")
new = np.array([user_input])
df = cv.transform(new)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
output = model.predict(df)
print(output) """
