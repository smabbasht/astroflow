import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from bnn import BayesianNN
from dataset import Dataset

tf.get_logger().setLevel('ERROR')

# ----------------------------------------------

print("------------------------------------")
print()
print("Making Data Class")
print()
dataset = Dataset('../blackligo-data-genie/data/', 'realization_0', 0.3)
print("Loading Data")
print()
# input_data = dataset.preprocess_data()
input_data = dataset.load_data("dataset_arrays")
print("Loading BayesianNN Model")
model = BayesianNN(input_data[0][0].shape, input_data)
model.load_model('models/model.pth')
model.eval()
# model.load_model('models/model.pth')
# predictions, truths = model.infer(data_offset=2000, n_examples=50)
# model.plot(predictions, truths, "figures", "cluster_run_02_final_plot")
# predictions, truths = model.infer(data_offset=2000, n_examples=500)
# model.plot(predictions, truths, "figures", "cluster_run_02_final_plot")

