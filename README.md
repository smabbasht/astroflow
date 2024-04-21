# Astroflow 🚀
A python library accompanied with a minimal dashboard to allow practicioners to easily infer Binary Blackholes population parameters using a mix of techniques, which involves normalizing flows, traditional neural network layers, Convolutional layers, Pooling layers and Bayesflow's DeepSet architecture. These techniques are jointly employed to produce an architecture that performs the inference of interest 

# Architecture
The architecture is broadly divided into two main subarchitectures; Summary Network and an Inference Network.

## Summary Network
Summary Network consists of traditional nerual net layers along with the summary layer of DeepSet provided by Bayesflow. This produces a summary vector which is then fed to our inference network.

## Inference Network
Inference Network exploits the summary vector received using coupling layers provided by the Bayesflow's InvertibleNetwork architecture. It produces the distribution object so you can easily produce any number of samples for the parameters of interest. 

## How to use
- Clone the repository.
```
git clone https://github.com/BlackLigo/astroflow/
```
- Run the following command to install dependencies: 
```
pip install -r requirements.txt
```
- Run `convert.py` script to convert nc files to csv (not required if you already have data in .csv files). Before running change the path to the location where your files are stored.
- Run `dataset.py` **with adjusted paths** to convert your csv files into `*.npy` files to be used by the model. (You can now delete your csv files to save space).
- Run `main.py` command to train and test model. Before running change the **datapath** variable to location where your data is stored.
