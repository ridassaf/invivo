from SmilesEnumerator import SmilesEnumerator
from sklearn.model_selection import train_test_split
import keras.backend as K
from SmilesEnumerator import SmilesIterator

import numpy as np
import pandas as pd

from sklearn.utils import resample


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

def decode(X): #decodes one-hot encoded data
    X_ret = np.empty([X.shape[0],X.shape[1]])
    for i in range(X.shape[0]):
        for j in range(X_.shape[1]):
            X_ret[i][j] = np.argmax(X[i][j])
    return X_ret

def createLSTM(input_shape):
    output_shape = 1
    
    model = Sequential()
    model.add(LSTM(64,input_shape=input_shape,dropout = 0.2))
    model.add(Dense(output_shape,activation="sigmoid"))
        
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001))
    print model.summary()

NN = false # set to true to run LSTM

targets  = ['target1', 'target2', 'target3','target4','target5','target6','target7','target8', 'target9','target10','target11','target12']
sme = SmilesEnumerator()
auc = []
f1 = []
d = pd.read_csv("data.csv")
for i in range(1,13): #to cover the 12 targets
	target = 'target' + str(i)
    d[target] = d.apply( # fill missing values with median of row
	    lambda row: np.median(row) if np.isnan(row[target]) else row[target],
	    axis=1
	)
	da = d.copy()
	#da.dropna(inplace=True) # alternatively, drop rows with missing values

	#downSample:
	df_majority = da[da[target]==0]
	df_minority = da[da[target]==1]
	df_majority_downsampled = resample(df_majority, 
					 replace=False,    # sample without replacement
					 n_samples=df_minority.shape[0],     # to match minority class
					 random_state=123) # reproducible results
	data = pd.concat([df_majority_downsampled, df_minority])

    # generate embeddings and convert dataset
	sme.fit(d['smiles']) #"smiles"
	sme.leftpad = True
	#The SmilesEnumerator must be fit to the entire dataset, so that all chars are registered
	targets.remove(target)
	generator = SmilesIterator(data['smiles'], data[target], sme, batch_size=data['smiles'].shape[0] ,dtype=K.floatx(), extras = data[others])
	X,y = generator.next()
	X_tr,  X_te, y_train, y_test = train_test_split(X, y, random_state=42)

    # decode one-hot encoded data
	X_train = decode(X_tr)

    if NN:
        model = createLSTM(X_train.shape[1:])
        model.fit(X_train,y_train, steps_per_epoch=100, epochs=5)
    else:
        model = RandomForestClassifier( n_estimators=100, max_depth=2, random_state=0)
        model.fit(X_train, y_train)
        
	X_test= decode(X_te)
	y_pred  = model.predict_proba(X_test)
    if not NN:
        y_pred = [p[1] for p in y_pred]
	auc.append(round(roc_auc_score(y_test, y_pred),2))
	y_pred = [round(p) for p in y_pred]
	f1.append(round(f1_score(y_test, y_pred),2))

print auc
print f1
print sum(auc)/len(auc)
print sum(f1)/len(sum)

