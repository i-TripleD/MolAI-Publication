
# MolAI: A Deep Learning Framework for Efficient Molecular Descriptor Generation and Advanced Drug Discovery Applications
Implementation of "MolAI" by S.J. Mahdizadeh and L.A. Eriksson.

## Getting Started

1. **Prerequisites:**
   * Python 3.8
   * TensorFlow 2.10
   * scikit-learn 1.0.2
   * XGBoost 1.5.2
   * RDKit 2022.9.4
   * Dimporphite-DL 1.3.2
   * pandas 1.3
   * tqdm 4.67

2. **Installation:**

```bash
# Clone this repository
git clone git@github.com:i-TripleD/MolAI-Publication.git

# Change directory
cd MolAI

# Install required packages
pip install -r requirements.txt

#Merge MolAI model
cat models_MolAI/Model_Trained_epoch_6_part_* > models_MolAI/Model_Trained_epoch_6.h5
```

## Usage

### Pre-Trained Models

Pre-trained MolAI models and encoding are available in the "models_MolAI" directory. You can use these for immediate predictions. Refer to the paper for model details and performance metrics.

The repository also contains pretrained models for iLP and 14 ADMET features. Training files for the ADMET features are supplied for demostrational use of the toolset. 

#### Retrain an ADMET Model

For demostational purposes, all files necessary to retrain the models are supplied in each ADMET features folder. In this example AMES is retrained with the supplied raw_data.csv:

```bash
# Change directory to AMES folder
cd AMES

# Generate and predcit the correct protonation state using iLP
python run_iLP.py

# Train the AMES model on the protonated SMILES
python train.py

```

## Citation

If you find MolAI useful in your research, please cite the following paper:

```
Mahdizadeh, S.J. and Eriksson, L.A. (2024). ...
```
