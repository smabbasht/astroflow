import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import save_model, load_model
from tensorflow.keras import Sequential
import tensorflow_probability as tfp 
tfd = tfp.distributions

from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from bayesflow.networks import DeepSet
from bayesflow.amortizers import AmortizedPosterior
# from forked_networks import DeepSet

class BayesianFlowNetwork:
    def __init__(self, _input_shape, input_data, _batch_size):
        self.input_shape = _input_shape
        self.prior = self.define_prior()
        self.X_train, self.X_test = self.convert_to_tensor(input_data[:2])
        self.y_train, self.y_test = self.convert_to_tensor(input_data[2:])
        self.trainer=None 
        self.summary_network=None 
        self.inference_network=None
        self.amortized_posterior=None
        self.batch_size=_batch_size

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
        self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
        # self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], -1))


    def dataset_info(self):
        print(f"X_train.shape = {self.X_train.shape}")
        print(f"y_train.shape = {self.y_train.shape}")
        print(f"X_test.shape = {self.X_test.shape}")
        print(f"y_test.shape = {self.y_test.shape}")

    def set_summary_network(self, _output_size=128):
        # self.summary_network = DeepSet(self.batch_size, len(self.X_train), self.input_shape)
        # dense_s1={
        #         "units": 128,
        #         "input_shape": self.input_shape
        # }
        self.summary_network = DeepSet(summary_dim=30)

    def set_inference_network(self, _num_params=3, _num_blocks=2, _subnet_constructor=Sequential):
        self.inference_network = InvertibleNetwork(num_params=_num_params)

    def set_amortized_posterior(self):
        self.amortized_posterior = AmortizedPosterior(self.inference_network, self.summary_network)

    def setup_trainer(self):
        self.trainer = Trainer(amortizer=self.amortized_posterior, generative_model=None, 
                               # checkpoint_path='models'
        )  

    def train(self, _batch_size=None, _epochs=100):
        if _batch_size==None:
            _batch_size=self.batch_size
        simulations_dict = {
            'sim_data': self.X_train,
            'prior_draws': self.y_train
        }
        self.trainer.train_offline(simulations_dict, batch_size=_batch_size, epochs=_epochs, 
                                   # optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                                   # save_checkpoint=True
        )


    def define_prior(self):
        priors = tfd.JointDistributionNamed({
            'alpha': tfd.Uniform(low=-5., high=5.),
            'mass_min': tfd.Uniform(low=5., high=20.),
            'mass_max': tfd.Uniform(low=30., high=200.)
        })
        return priors

    def predict(self, X=None):
        # if X is None:
        #     return self.amortized_posterior.sample({'sim_data': self.X_test}, n_samples=1000)
        # X = np.reshape(X, (X.shape[0], X.shape[1], -1))
        # X=self.convert_to_tensor(X)[0]
        simulations_dict={'summary_conditions': self.X_test[5:15]}
        print()
        print(f"expected_means: \n{self.y_test[5:15]}")
        print()
        print("predicted_means:")

        return self.amortized_posterior.sample(simulations_dict, 1000, to_numpy=True).mean(axis=1)
        # return self.amortized_posterior.sample({'sim_data': X}, n_samples=1000)

    def predict_2(self, _batch_size=None, _epochs=100):
            if _batch_size==None:
                _batch_size=self.batch_size
            simulations_dict = {
                'sim_data': self.X_train[0],
                'prior_draws': self.y_train[0]
            }
            return self.amortized_posterior(simulations_dict)
            # self.trainer.train_offline(simulations_dict, batch_size=_batch_size, epochs=_epochs, 
            #                            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            #                            # save_checkpoint=True
            # )

    def save_model(self, filepath):
        self.amortized_posterior.summary_net.save(filepath+'_summary_net')
        self.amortized_posterior.inference_net.save(filepath+'_inference_net')
        # save_model(self.amortized_posterior.summary_net, filepath + '_summary_net')
        # save_model(self.amortized_posterior.inference_net, filepath + '_inference_net')

    def load_model(self, filepath):
        self.summary_network = load_model(filepath + '_summary_net')
        self.inference_network = load_model(filepath + '_inference_net')
        self.set_amortized_posterior()

    def convert_to_tensor(self, data):
        return [tf.convert_to_tensor(d, dtype=tf.float32) for d in data]

    # def define_prior(batch_size):
    #     alpha = np.random.uniform(low=-5.0, high=5.0, size=(batch_size, 1))
    #     mass_min = np.random.uniform(low=5.0, high=20.0, size=(batch_size, 1))
    #     mass_max = np.random.uniform(low=30.0, high=200.0, size=(batch_size, 1))
    #     return np.concatenate([alpha, mass_min, mass_max], axis=1)
    # def predict(self, X):
    #     return self.trainer.sample_posterior(X, n_samples=1000).numpy()
    #
    # def eval(self, X_test=self.X_test):
    #     posterior_samples = self.predict(X_test)
    #     print("Posterior Samples from X_test:", posterior_samples)
    #     return posterior_samples
    #
    #
    # def save_model(self, savepath="models/bfn.keras"):
    #     save_model(self.forward_model, savepath, overwrite=True)
    #     # self.forward_model.save("models/bfn.keras")
    #
    # def load_model(self, loadpath="models/bfn.keras"):
    #     self.forward_model = load_model(loadpath)


#     def build_forward_model(self):
#         return InvertibleNetwork(output_dim=3, hidden_shapes=[128, 64, 16])
#
#     def setup_trainer(self):
#         return ParameterEstimationTrainer(self.forward_model, self.prior)
#
#     def train(self, n_epochs=100, batch_size=128):
#         self.trainer.train(self.X_train, self.y_train, epochs=n_epochs, batch_size=batch_size)
#




# import tensorflow as tf
# from bayesflow.networks import InvertibleNetwork
# from bayesflow.trainers import ParameterEstimationTrainer
#
# # Assume your data is prepared and loaded into these variables
# X_train, y_train = None, None  # Your training data and labels
# X_test, y_test = None, None  # Your testing data and labels
#
# # Define an invertible network for the forward model
# forward_model = InvertibleNetwork(
#     output_dim=3,  # Assuming 3 parameters: alpha, mass_min, mass_max
#     hidden_shapes=[128, 64, 16],  # Hidden layer sizes
# )
#
# # Assuming a simple prior over parameters for demonstration
# def prior(batch_size):
#     # Assuming alpha, mass_min, and mass_max have specific prior distributions
#     # These should be replaced with distributions appropriate for your parameters
#     priors = tfd.JointDistributionNamed({
#         'alpha': tfd.Normal(loc=0., scale=1.),  # Example prior for alpha
#         'mass_min': tfd.Uniform(low=0., high=50.),  # Example prior for mass_min
#         'mass_max': tfd.Uniform(low=50., high=100.)  # Example prior for mass_max
#     })
#     return priors.sample(batch_size)
#
# # Setup the trainer for parameter estimation
# trainer = ParameterEstimationTrainer(forward_model, prior)
#
# # Training the model
# trainer.train(X_train, y_train, epochs=100, batch_size=128)
#
# # To generate samples for the test data
# posterior_samples = trainer.sample_posterior(X_test, n_samples=1000).numpy()
#
# # posterior_samples will contain the generated samples for each label parameter
#
#
