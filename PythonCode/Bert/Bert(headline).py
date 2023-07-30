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
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
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
import sys, os
from transformers import BertTokenizer
from textwrap3 import wrap
import random
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import numpy as np
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score
import torch

model_type = ""

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

print("Loading BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

path = '/home/adbuls/Title-Data/'
#path = '/home/adbuls/Final-labelled-data/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
print(len(csv_files))
o = pd.DataFrame(columns =["file","model","Accuracy","Precision","Recall","F1","input"])


model2 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model3 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

for f in csv_files:
    filename = f.split("/")[-1]
    print(filename)
    modelname = ""
    inputname = ""
    q = pd.DataFrame(columns =["file","model","Accuracy","Precision","Recall","F1","input"])
    data = pd.read_csv(f)


    Features = data.filter(['title','allinferences','heads-senti','allinferences-senti','class','senti-str-headline'], axis=1)
    #Features['heads-senti']= Features['heads-senti'].fillna(0)
    #Features['allinferences-senti']= Features['allinferences-senti'].fillna(0)
    #Features['allinferences']= Features['allinferences'].fillna("")
    Features['title']= Features['title'].fillna("")
    Features['senti-str-headline']= Features['senti-str-headline'].fillna("")
    Features['allinferences']= Features['allinferences'].fillna("")
    Features['class']= Features['class'].fillna("")
    print("im here")
    print(len(Features))    
    Features['title'] = Features['title'].dropna("").reset_index(drop=True)
    print(len(Features)) 
    Features['senti-str-headline'] = Features['senti-str-headline'].dropna("").reset_index(drop=True)
    print(len(Features)) 
    Features['allinferences'] = Features['allinferences'].dropna("").reset_index(drop=True)
    print(len(Features)) 
    Features['class'] = Features['class'].dropna("").reset_index(drop=True)
    Features['class'].dropna("")
    print(len(Features)) 
    print("im here")
    print(len(Features)) 
    Features.dropna(inplace=True, subset=['class'])
    Features = Features[pd.notnull(Features['class'])]

    Features = Features[Features['class'] != None]
    Features = Features[Features['class'].notna()]
    index_names = Features[ Features['class'] == ""].index
    Features.drop(index_names, inplace = True)
    Features.reset_index(drop=True, inplace=True)
    
    index_names2 = Features[ Features['title'] == ""].index
    Features.drop(index_names2, inplace = True)
    Features.reset_index(drop=True, inplace=True)
 
    
    if len(Features["class"].unique()) == 2:
        model = model2
        
    elif len(Features["class"].unique()) == 3:
        model = model3

    Features['title'] = Features["title"].astype(str) + Features["allinferences"].astype(str)
    Features['title'] = Features["title"].astype(str) + Features["senti-str-headline"].astype(str)
    stop_words_l= stopwords.words('english')
    Features['title'].dropna().apply(lambda x: [item for item in x if item not in stop_words_l])
    Features['title'] = Features['title'].dropna().map(lambda x: x.lower())
    #Features['allinferences'] .apply(lambda x: [item for item in x if item not in stop_words_l])
    #Features['allinferences'] = Features['allinferences'].map(lambda x: x.lower())
    print(len(Features)) 
    print("im here")
    print(len(Features))
    print("classes")
    print(Features["class"].unique())
    print(len(Features["class"].unique()))
    print(Features['class'].value_counts())
    train = Features.filter(['title'])
    test  = Features.filter(['class'])
    print("features")
    print(len(train))
    print(len(test))
    classes = label_encoder.fit_transform(test)
    X_train, X_test, y_train, y_test = train_test_split(train, classes.ravel(),test_size=0.2, shuffle = True, random_state = 8,stratify=classes)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train.ravel(), test_size=0.25, random_state= 8,stratify=y_train)
    Features = Features.filter(['title','class'])
    print("im here")
    print(len(Features))
    try:
        sen_w_feats = []
        labels2 = []
        labe = []
        for index, row in Features.iterrows():
            sen_w_feats.append(row['title'])
            labels2.append(row['class'])
        labe = label_encoder.fit_transform(test)
        print(wrap(sen_w_feats[1], 80))
        batch_size = 16
        learning_rate = 1e-5
        epochs = 3
        max_len = 0
        for sent in sen_w_feats:
            input_ids = tokenizer.encode(str(sent), add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        print(max_len)
        trs = int(0.8*len(Features))
        vas = int(0.1*len(Features))
        tes = len(Features)-(trs+vas)
        assert((trs+vas+tes) == len(Features))
        indeces = np.arange(0, len(Features))
        print(len(Features))
        print(len(indeces))
        random.shuffle(indeces)
        print(indeces.min(axis=0))
        print(indeces.max(axis=0))
        tr_idx  = indeces[0:trs]
        val_idx = indeces[trs:(trs+vas)]
        tes_idx = indeces[(trs+vas):]
        print(trs)
        print(vas)
        print(tes)
        print(max_len)
        print(len(sen_w_feats))
        max_len= max_len
        input_ids = []
        attention_masks = []
        for sent in sen_w_feats:
            encoded_dict = tokenizer.encode_plus(str(sent), add_special_tokens=max_len, truncation= True, padding= 'max_length', return_attention_mask=True, return_tensors='pt',) 
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids       = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labe)
        labels = labels.type(torch.LongTensor)
        print(len(input_ids))
        print(len(attention_masks))
        print(len(labels))
        tr_ds = TensorDataset(input_ids[tr_idx], attention_masks[tr_idx], labels[tr_idx])
        va_ds = TensorDataset(input_ids[val_idx], attention_masks[val_idx], labels[val_idx])
        te_ds = TensorDataset(input_ids[tes_idx], attention_masks[tes_idx], labels[tes_idx])
        train_dataloader = DataLoader(tr_ds, sampler=RandomSampler(tr_ds))
        validation_dataloader = DataLoader(va_ds, sampler=RandomSampler(va_ds))
        test_dataloader = DataLoader(te_ds, sampler=RandomSampler(te_ds))
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps = 1e-8)
        to_steps  = len(train_dataloader)*epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=to_steps)

        device = torch.device("cpu")
        s_val = 42
        random.seed(s_val)
        np.random.seed(s_val)
        torch.manual_seed(s_val)
        torch.cuda.manual_seed_all(s_val)
        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()        
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
                loss = result.loss
                logits = result.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            print("")
            print("Running Validation...")
            t0 = time.time()
            model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                with torch.no_grad():        
                    result = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels,return_dict=True)
                loss = result.loss
                logits = result.logits
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
            training_stats.append({'epoch': epoch_i + 1,'Training Loss': avg_train_loss,'Valid. Loss': avg_val_loss, 'Valid. Accur.': avg_val_accuracy,'Training Time': training_time,'Validation Time': validation_time})
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        model.eval()
        predictions , true_labels = [], [] 
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,return_dict=True)
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.append(logits)
            true_labels.append(label_ids)
        flat_predictions = np.concatenate(predictions, axis=0)
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = np.concatenate(true_labels, axis=0)
    
        f1 = f1_score(flat_true_labels, flat_predictions, average="macro")
        acs =accuracy_score(flat_true_labels, flat_predictions)
        f1 =f1_score(flat_true_labels, flat_predictions, average="macro")
        re =recall_score(flat_true_labels, flat_predictions, average="macro")
        pr =precision_score(flat_true_labels, flat_predictions, average="macro")
        new_row = {"file":filename,"model":"BERT","Accuracy":str(round(acs,2)),"Precision":str(round(pr,2)),"Recall":str(round(re,2)),"F1":str(round(f1,2))}
        o = o.append(new_row,ignore_index=True)
        q = q.append(new_row,ignore_index=True)
        q.to_csv(os.path.join(filename))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(e))
o.to_csv(os.path.join("PM-BERT-Headlines.csv"))