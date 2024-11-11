'''
	ml TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 

''' 

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tqdm import tqdm
from keras.layers import Input, LSTM, Dense, Concatenate
import pickle
import time
from rdkit import Chem
from tqdm import tqdm
import contextlib
import multiprocessing
from dimorphite_dl import DimorphiteDL
import re

start1 = time.time()

# convert random SMILES to canonical SMILES

def generate_canonical_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return None 

if __name__ == "__main__":

    input_file = "raw_data.csv"
    output_file = "SMILES_canonical.csv"
    chunksize = 1e6

    pool = multiprocessing.Pool()
    with open('out_canonicalizer.txt', 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        with open(output_file, 'w') as f:
            first_append = True


            for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize, header=None)):
                nu_smiles = 0
                input_smiles = chunk[1].tolist()
                idxs = chunk[2].values

                results = pool.imap(generate_canonical_smiles, input_smiles)

                for i, sm in enumerate(results):
                    target = idxs[i]
                    if first_append:
                        f.write(f"{sm},{target}\n")
                    else:
                        f.write(f"\n{sm},{target}")
            
                first_append = False
                nu_smiles = nu_smiles + i

        end = time.time()
        execution_time = end - start1
        print(nu_smiles, "molecules have been canonicalized in", "%.1f" %execution_time, "seconds")


# run Dimorphite-dl to generate all protonation states at pH=7.0 +/- 2.0

start2 = time.time()

def generate_protonations(smile):
    return dimorphite_dl.protonate(smile)

if __name__ == "__main__":


    dimorphite_dl = DimorphiteDL(
        min_ph=5.0,
        max_ph=9.0,
        max_variants=20,
        label_states=False,
        pka_precision=1.0
    )

    input_file = "SMILES_canonical.csv"
    output_file = "dimorph_canonical.csv"
    chunksize = 100000

    pool = multiprocessing.Pool()
    with open('out_dimorphite.txt', 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        with open(output_file, 'w') as f:
            first_append = True
            j=0
            for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize, header=None)):
                input_smiles = chunk[0].values
                idxs = chunk[1].values

                results = pool.imap(generate_protonations, input_smiles)
                for i, protonation_states in enumerate(results):
                    smile = input_smiles[i]
                    target = idxs[i]

                    for protonation_state in protonation_states:
                        if first_append:
                            f.write(f"{protonation_state},{target},{i*(j+1)}\n")
                        else:
                            f.write(f"\n{protonation_state},{target},{i*(j+1)}")

                first_append = False
            j=j+1

        end = time.time()
        execution_time = end - start2
        print(nu_smiles, "protonation states have been generated in", "%.1f" %execution_time, "seconds")



# run MolAI to generate the latent space for all protonation states

dimorph = pd.read_csv("dimorph_canonical.csv", header=None)

filtered_rows = dimorph.replace(to_replace=['Cl','Br'], value=['X','Y'], regex=True)
#filtered_rows = filtered_rows[~filtered_rows[0].str.contains('B|Si|10|\.')]



allowed_characters = {'c','3','5','=','O','+','8','[','s',']','C','n','p','H','%','-','(','I','1','4','Y','X','6','!','S',')','#','o','2','P','N','7','F'}
escaped_characters = ''.join(re.escape(char) for char in allowed_characters)

regex = f"[^{escaped_characters}]"


filtered_rows = filtered_rows[~filtered_rows[0].str.contains(regex)]


filtered_rows = filtered_rows[filtered_rows[0].apply(len) < 110]
filtered_rows = filtered_rows[filtered_rows[0].apply(len) > 2]
filtered_rows.reset_index(inplace=True, drop=True)


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
        
        one_hot[i, 0, char_to_int["!"]] = 1
        
        for j, c in enumerate(smile):
            if c in char_to_int:
                one_hot[i, j+1, char_to_int[c]] = 1
            else:
                continue  # Skip if char not in char_to_int
                
        one_hot[i, len(smile)+1, char_to_int["$"]] = 1
        one_hot[i, len(smile)+2:, char_to_int["%"]] = 1
        
    return one_hot[:, 1:, :], one_hot[:, :-1, :]


X_test, _ = vectorize(filtered_rows[0].values)

batch_size = 1e6


num_batches = np.ceil(X_test.shape[0] / batch_size).astype(int)
predictions = []

for batch_idx in range(num_batches):
    start_idx = int(batch_idx * batch_size)
    end_idx = int(min((batch_idx + 1) * batch_size, X_test.shape[0]))
    
    X_test_batch = X_test[start_idx:end_idx]
    batch_predictions = smiles_to_latent_model.predict(X_test_batch)
    predictions.append(batch_predictions)

latent = np.concatenate(predictions, axis=0)


# Run iPL to pinpoint the dominant protonation state at the given pH

models = list()       
for i in range(5):   
    filename = '../models_iLP/model_1_' + str(i+1)
    model = load_model("%s.h5" %filename) 
    models.append(model)

prob = [model.predict(latent).flatten() for model in models]
prob_mean = np.array(prob).mean(axis=0)

res = pd.DataFrame(prob_mean)
res[1] = filtered_rows[0]
res[2] = filtered_rows[1]
res[3] = filtered_rows[2]

max_prob = res.groupby(3)[0].transform('max')
res[4] = np.where(res[0] == max_prob, 1, 0)   #only the highest probebility

'''diff = 0.03
res[3] = np.where(max_prob - res[0] <= diff, 1, 0)'''  #the highest probebility plus all in the range 'diff'

res = res[res[4] != 0]
res = res.replace(to_replace=['X','Y'], value=['Cl','Br'], regex=True)
res.drop([0,3,4], axis=1, inplace=True)
res.rename(columns={1: 'prepped_SMILES', 2: 'target'}, inplace=True)

res.to_csv('prepped_SMILES.csv', index=None)




end = time.time()
execution_time = end - start1
print(nu_smiles, "molecules have been prepared in", "%.1f" %execution_time, "seconds")



