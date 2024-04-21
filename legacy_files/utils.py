# import tensorflow as tf
# import tensorflow_probability as tfp
# tfpl = tfp.layers
# tfd = tfp.distributions
#
# # Set the event shape
# event_shape = (3,)
#
# # Get the total number of parameters needed for the IndependentNormal distribution
# params_size = tfpl.IndependentNormal.params_size(event_shape)
#
# # Create the IndependentNormal distribution
# distribution = tfpl.IndependentNormal(params_size, convert_to_tensor_fn=tfp.distributions.Distribution.mean)
#
# # Assuming you want to sample from the distribution
# samples = distribution.sample()
#
# # Print the samples
# print(samples)
#
#
# def fib(n):
import numpy as np
np.save("dataset_arrays_3000/X_train.npy", np.load("dataset_arrays_5/X_train.npy")[:3000])
np.save("dataset_arrays_3000/X_test.npy", np.load("dataset_arrays_5/X_test.npy")[:100])
np.save("dataset_arrays_3000/y_train.npy", np.load("dataset_arrays_5/y_train.npy")[:3000])
np.save("dataset_arrays_3000/y_test.npy", np.load("dataset_arrays_5/y_test.npy")[:100])

