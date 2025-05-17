# MUSDTA
MUSDTA is an advanced computational framework for predicting drug–target binding affinity (DTA). By integrating multimodal feature fusion and structural modeling, it comprehensively captures the complex interactions between drugs and targets, aiming to enhance the efficiency and accuracy of drug discovery.

# Dependency
The directory of Python packages required for the dependencies should be placed in a requirements file.

# Data preparation
The MUSDTA data is downloaded from https://...

The Pertrainmodel is downloaded from https://...

1.Unpacking data.zip

2.Unpacking Pertrainmodel.zip

    /data/davis
    /data/kiba
    /data/pdbbind
    /data/bindingdb

    /Pertrainmodel/ESM2_t36_3B_UR50D
    /Pertrainmodel/ChemBERTa

# Running
    python data_pre_process.py

    python run.py
           
