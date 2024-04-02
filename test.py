import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

# Set the event shape
event_shape = (3,)

# Get the total number of parameters needed for the IndependentNormal distribution
params_size = tfpl.IndependentNormal.params_size(event_shape)

# Create the IndependentNormal distribution
distribution = tfpl.IndependentNormal(params_size, convert_to_tensor_fn=tfp.distributions.Distribution.mean)

# Assuming you want to sample from the distribution
samples = distribution.sample()

# Print the samples
print(samples)


def fib(n):

