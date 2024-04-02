# bnn
A library to infer Binary Blackholes population using Bayesian Neural Networks

# MCNeural Networks
A simple Neural Network with MCDropout to approximately simulate Bayesian Neural Networks implemented in Model.py file

## How to use
- clone the repository.
- run the following command to install dependencies `pip install -r requirements.txt`.
- run `convert.py` script to convert nc files to csv (not required if you already have data in .csv files). Before running change the path to the location where your files are stored.
- run the `python3 test.py` command to train and test model. Before running change the **datapath** variable to location where your data is stored.
