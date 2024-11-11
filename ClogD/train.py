'''
ml scikit-learn/1.0.1-foss-2021b
ml TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 
'''

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dropout, Layer
import pickle
from pickle import dump
from pickle import load

from tqdm import tqdm
from tensorflow.keras import layers, models
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.callbacks import History, Callback, LearningRateScheduler

data = pd.read_csv('prepped_SMILES.csv').dropna().drop_duplicates(keep='first')
data

import re
smiles_repl = data.replace(to_replace=['Cl','Br'], value=['X','Y'], regex=True)



filtered_rows = smiles_repl[~smiles_repl['prepped_SMILES'].str.contains('B|Si|10')]

allowed_characters = {'c', '3', '5', '=', 'O', '+', '8', '[', 's', ']', 'C', 'n', 'p', 'H', '%', '-', '(', 'I', '1', '4', 'Y', 'X', '6', '!', 'S', ')', '#', 'o', '2', 'P', 'N', '7', 'F'}
escaped_characters = ''.join(re.escape(char) for char in allowed_characters)
regex = f"[^{escaped_characters}]"
filtered_rows = filtered_rows[~filtered_rows['prepped_SMILES'].str.contains(regex)]

filtered_rows = filtered_rows[filtered_rows['prepped_SMILES'].apply(len) < 111]
filtered_rows.reset_index(inplace=True)


with open('../models_MolAI/char_to_int.pkl', 'rb') as f:
    char_to_int = pickle.load(f)
with open('../models_MolAI/int_to_char.pkl', 'rb') as f:
    int_to_char = pickle.load(f)
    
char_to_int_len = len(char_to_int)
max_smi_len = 111

smiles_to_latent_model = load_model('../models_MolAI/smi2lat_epoch_6.h5')

def vectorize(smiles):
    one_hot = np.zeros((smiles.shape[0], max_smi_len, char_to_int_len), dtype=np.int8)
    for i, smile in tqdm(enumerate(smiles)):
        
        # Encode the start char
        one_hot[i, 0, char_to_int["!"]] = 1
        
        # Encode the rest of the chars
        for j, c in enumerate(smile):
            if c in char_to_int:
                one_hot[i, j+1, char_to_int[c]] = 1
            else:
                continue  # Skip if char not in char_to_int
                
        # Encode end char
        one_hot[i, len(smile)+1, char_to_int["$"]] = 1
        
        one_hot[i, len(smile)+2:, char_to_int["%"]] = 1
        
    # Return two, one for input and the other for output
    return one_hot[:, 1:, :], one_hot[:, :-1, :]


X_test, Y_test = vectorize(filtered_rows['prepped_SMILES'].values)

from time import time
start = time()
latent = smiles_to_latent_model.predict(X_test)
end = time()
print(end-start)

np.save('latent.npy', latent)

train = pd.DataFrame(latent)
train['Target'] = filtered_rows['target']

x_train = train.drop("Target",axis=1)
y_train = train.pop('Target')  

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

def scheduler(epoch, lr):
    return lr * 0.9  
lr_scheduler = LearningRateScheduler(scheduler)


MAEt = []      #for the first metric (MAE)
MSEt = []     #for the first metric (RMSE)
St = []        #for the first metric (Spearman)
Pt = []        #for the first metric (Pearson)
folds = 5     #define the number of kfolds

for i in range(1):
    i+=1
    kf = KFold (folds, shuffle=True, random_state=i)     #kfold set up
    fold = 0
    
    for train, test in kf.split(x_train, y_train):     #train
        fold+=1
        print(f"Fold #{i}_{fold}")

        
        # define and fit model
        DNN_model = tf.keras.Sequential()
        DNN_model.add(tf.keras.layers.Dense(2048, input_shape=(512,), activation='relu', kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(1024, activation='relu',kernel_initializer='HeNormal'))
        #DNN_model.add(tf.keras.layers.Dense(50, activation='relu',kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(1, activation='linear',kernel_initializer='HeNormal'))

        DNN_model.compile(optimizer=tf.optimizers.Adam(0.0001), loss='mean_absolute_error', metrics=['mse'])
        
        early_stopping = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True, monitor='val_loss')
        
        history = DNN_model.fit(x_train[train], y_train[train], epochs=100, validation_data=(x_train[test], y_train[test]), callbacks=[early_stopping, lr_scheduler], verbose=1, batch_size = 4)
        
        pred = DNN_model.predict(x_train[test])
        pred = np.squeeze(np.array(pred))
        # save model to file

        filename  = './models/model_' + str(i) + '_' + str(fold) + '.h5'
        DNN_model.save(filename )
        print('Saved: %s' % filename )
        
        pd.concat([pd.DataFrame(y_train[test]), pd.DataFrame(pred)], axis=1, ignore_index=True).to_csv("%s.csv" %filename, header=None)


        # calculate metrics and statistics

        scores = DNN_model.evaluate(x_train[test], y_train[test], verbose=0)

        print("%s: %.2f" % (DNN_model.metrics_names[0], scores[0]))
        MAEt.append(scores[0])
        
        print("%s: %.2f" % (DNN_model.metrics_names[1], scores[1]))
        MSEt.append(scores[1])
        
        corr1, _ = spearmanr(y_train[test], pred)
        print('Spearmans correlation: %.3f' % corr1)
        St.append(corr1)

        corr2, _ = pearsonr(y_train[test], pred)
        print('Pearsons correlation: %.3f' % corr2)
        Pt.append(corr2)
        print('----------------------------------------------------')

    
print("Average_MAE = %.2f (+/- %.2f)" % (np.mean( MAEt), np.std( MAEt)))
print("Average_MSE = %.2f (+/- %.2f)" % (np.mean(MSEt), np.std(MSEt)))
print("Average_Spearmans = %.2f (+/- %.2f)" % (np.mean(St), np.std(St))) 
print("Average_Pearsons = %.2f (+/- %.2f)" % (np.mean(Pt), np.std(Pt)))