#! /bin/bash

# Setting up the environment
sudo apt install -y python3-pip
pip3 install -r requirements.txt

# Generating the Dataset
python3 data_generation/data_gen.py

# Running the model 
python3 main.py

# Zipping the models and plots for easier transfer
zip -r models.zip models
zip -r plots.zip figures

cp models.zip ../
cp plots.zip ../

