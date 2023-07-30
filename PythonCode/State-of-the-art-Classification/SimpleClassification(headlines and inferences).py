import os
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

label_encoder = LabelEncoder()




path = '/home/adbuls/Title-Data/'
#path = '/home/adbuls/sample/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
print(len(csv_files))
print(len(csv_files))
o = pd.DataFrame(columns =["file","model","Accuracy","Precision","Recall","F1","input"])

counter = 0
for f in csv_files:
    filename = str(f.split("/")[-1])
    q = pd.DataFrame(columns =["file","model","Accuracy","Precision","Recall","F1","input"])
    print("filename")
    print(filename)
    modelname = ""
    inputname = ""
    data = pd.read_csv(f)
    Features = data.filter(['title','body','cons','Label'], axis=1)
    
    #Features['title']= Features['title'].fillna(0)
    #Features['body']= Features['body'].fillna(0)
    Features['cons']= Features['cons'].fillna("")
    Features['cons'] = Features.cons.apply(lambda x: (x.replace('-', ' ')).split("___"))
    Features['cons'] = [' '.join(map(str, l)) for l in Features['cons']]
    
    Features['title']= Features['title'].fillna("")
    Features['body']= Features['body'].fillna("")
    stop_words_l= stopwords.words('english')
    
    ##stop_words_l= stopwords.words('english')
    #F#eatures['title']=Features.title.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split(' ') if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) 
    #Features['body'] = Features.body.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split(' ') if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) 
    
    #Features['title'] = Features['title'].map(lambda x: x.lower())
    #Features['body'] = Features['body'].map(lambda x: x.lower())
    
    #Features['title'] .apply(lambda x: [item for item in x if item not in stop_words_l])
    #Features['body'] .apply(lambda x: [item for item in x if item not in stop_words_l])

    
    #Features['title'] = Features['title'].map(lambda x: x.lower())
    #Features['body'] = Features['body'].map(lambda x: x.lower())
    
    train = Features.filter(['title','body','cons'], axis=1)
    test  = Features.filter(['Label'], axis=1)
    
    classes = label_encoder.fit_transform(test)
    try:
        X_train, X_test, y_train, y_test = train_test_split(train, classes,test_size=0.2, shuffle = True, random_state = 8,stratify=classes)
        X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state= 8,stratify=y_train)
    
    
        get_title   = FunctionTransformer(lambda x: x['title'], validate=False)
        get_body    = FunctionTransformer(lambda x: x['body'], validate=False)
        get_cons    = FunctionTransformer(lambda x: x['cons'], validate=False)

        title  = Pipeline([('selector',get_title),('vectorizer', CountVectorizer())])
        body  = Pipeline([('selector',get_body),('vectorizer', CountVectorizer())])
        cons  = Pipeline([('selector',get_cons),('vectorizer', CountVectorizer())])
        
        for i in range(6):
            if i == 0:
                union = FeatureUnion([('title',title)])
                inputname = "Title"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                counter = counter + 1;
            elif i == 1:
                union = FeatureUnion([('body',body)])
                inputname = "Body"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                    counter = counter + 1
            elif i == 2:
                union = FeatureUnion([('cons',cons)])
                inputname = "Cons"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                    counter = counter + 1  
            elif i == 3:
                union = FeatureUnion([('title',title),('body',body)])
                inputname = "Title+Body"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                    counter = counter + 1
            elif i == 4:
                union = FeatureUnion([('title',title),('cons',cons)])
                inputname = "Title+Cons"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                    counter = counter + 1
            else:
                union = FeatureUnion([('title',title),('body',body),('cons',cons)])
                inputname = "Title+Body+Cons"
                for j in range(5):
                    if j == 0:
                        pl = Pipeline([('union',union),('clf', LogisticRegression())])
                        modelname = "LogisticRegression"
                    elif j == 1:
                        pl = Pipeline([('union',union),('clf', RandomForestClassifier())])
                        modelname = "RandomForestClassifier"
                    elif j == 2:
                        pl = Pipeline([('union',union),('clf', DecisionTreeClassifier())])
                        modelname = "DecisionTreeClassifier"
                    elif j == 3:
                        pl = Pipeline([('union',union),('clf', KNeighborsClassifier())])
                        modelname = "KNeighborsClassifier"
                    else:
                        pl = Pipeline([('union',union),('clf', SVC())])
                        modelname = "SVC"
                    pl.fit(X_train, y_train)
                    prediction = pl.predict(X_test)
                    print(inputname)
                    print(modelname)
                    acs =accuracy_score(y_test, prediction)
                    f1 =f1_score(y_test, prediction, average="macro")
                    re =recall_score(y_test, prediction, average="macro")
                    pr =precision_score(y_test, prediction, average="macro")
                    new_row = {"file":filename,"model":modelname,"Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2)),"input":inputname}
                    o = o.append(new_row,ignore_index=True)
                    q = q.append(new_row,ignore_index=True)
                    j = j + 1
                    print(counter);
                    counter = counter + 1
        i = i + 1
        print(inputname)
        q.to_csv(filename)
    except:
        z=0
o.to_csv("Barrier-Classification-Results-.csv")
