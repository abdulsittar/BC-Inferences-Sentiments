from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from keras.layers import Activation, Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

label_encoder = LabelEncoder()



path = '/home/adbuls/Title-Data/'
#path = '/home/adbuls/Final-labelled-data/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
print(len(csv_files))
o = pd.DataFrame(columns =["file","model","Accuracy","Precision","Recall","F1","input"])

for f in csv_files:
    filename = f.split("\\")[-1]
    print(filename)
    modelname = ""
    inputname = ""
    data = pd.read_csv(f)
    Features = data.filter(['title','allinferences','heads-senti','allinferences-senti','class','senti-str-headline'], axis=1)
    stop_words_l= stopwords.words('english')
    Features['title']= Features['title'].fillna("")
    Features['senti-str-headline']= Features['senti-str-headline'].fillna("")
    Features['allinferences']= Features['allinferences'].fillna("")
    Features['class']= Features['class'].fillna("")
    x = Features['title'].values
    y = Features['class'].values
    #print(x)
    classes = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test =  train_test_split(x, classes, test_size=0.2, random_state=42)
    tokenizer = Tokenizer(num_words=10000)
    texts = pd.concat([train_df["title"],  train_df["allinferences"]])
    tokenizer.fit_on_texts(texts)
    vocab_size=len(tokenizer.word_index)+1
    xtrain= tokenizer.texts_to_sequences(x_train)
    xtest= tokenizer.texts_to_sequences(x_test) 
    maxlen=200
    xtrain=pad_sequences(xtrain,padding='post', maxlen=maxlen)
    xtest =pad_sequences(xtest, padding='post', maxlen=maxlen)
    embedding_dim=100
    #print(vocab_size)
    #print(classes)
    model=Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.LSTM(units=50,return_sequences=True))
    model.add(layers.LSTM(units=10))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8))
    
    if len(data["Label"].unique()) == 2:
        #print("2 class")
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        
        
    elif len(data["Label"].unique()) == 3:
        #print("3 class")
        model.add(Dense(1, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model.summary()
    model.fit(xtrain,y_train, epochs=10, batch_size=64, verbose=False)
    loss, acc = model.evaluate(xtrain, y_train, verbose=False)
    print("Training Accuracy: ", acc)

    loss, acc = model.evaluate(xtest, y_test, verbose=False)
    print("Test Accuracy: ", acc)
    prediction=model.predict(xtest)
    test_pred = []
    for a in range(len(prediction)):
        test_pred.append(round(round(prediction[a][0])))

    acs =accuracy_score(y_test, test_pred)
    f1 =f1_score(y_test, test_pred, average="macro")
    re =recall_score(y_test, test_pred, average="macro")
    pr =precision_score(y_test, test_pred, average="macro")
    new_row = {"file":filename,"model":"LSTM","Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2))}
    o = o.append(new_row,ignore_index=True)
o.to_csv(os.path.join("LSTM-Barrier-Classification-Results.csv"))
