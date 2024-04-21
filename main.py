import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from bnn import BayesianNN
from bfn import BayesianFlowNetwork
from dataset import Dataset
tf.get_logger().setLevel('ERROR')
# tf.compat.v1.disable_eager_execution()

# ----------------------------------------------

print("------------------------------------")
print()
print("Making Data Class")
print()
dataset = Dataset('../amanda/test/output_csv/', "", 0.1)
print("Loading Data")
print()
# input_data = dataset.preprocess_amanda('dataset_arrays_1000')
input_data = dataset.load_data("dataset_arrays_4000")
# print("Loaded 5x data")
# cropped_input_data = [x[:1000] for x in input_data[:2]], [x[:50] for x in input_data[2:]]
# del x for x in input_data

# ----------------------------------------------

print("Building BayesianNN Model")
# model = BayesianNN(input_data[0][0].shape, input_data, 64)
# model = BayesianFlowNetwork(cropped_input_data[0][0].shape, cropped_input_data, 64)
model = BayesianFlowNetwork(input_data[0][0].shape, input_data, 64)
model.dataset_info()
model.normalize_dataset()
# model.set_summary_network()
# model.set_inference_network()
# model.set_amortized_posterior()
# model.setup_trainer()
# model.setup_model()

# ----------------------------------------------

print("Started Training")
model.train()

# ----------------------------------------------

samples=model.predict()
model.plot(samples, model.y_test[:100], "figures", "bfn_14")

# ----------------------------------------------

# print("Saving the model")
# model.save_model('models/model')








# model.load_model('models/model.pth')
# predictions, truths = model.infer(data_offset=2000, n_examples=50)
# model.plot(predictions, truths, "figures", "cluster_run_02_final_plot")
# predictions, truths = model.infer(data_offset=2000, n_examples=500)
# model.plot(predictions, truths, "figures", "cluster_run_02_final_plot")
