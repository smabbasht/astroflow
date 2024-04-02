import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization, Dropout
from keras.layers import InputLayer
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from bayesflow.networks import DeepSet, SequenceNetwork, InvariantNetwork
from bayesflow.amortizers import AmortizedPosterior

class BayesianFlowNetwork:
    def __init__(self, input_shape, input_data, batch_size):
        self.input_shape = input_shape
        self.X_train, self.X_test = self.convert_to_tensor(input_data[:2])
        self.y_train, self.y_test = self.convert_to_tensor(input_data[2:])
        self.batch_size = batch_size

        self.setup_model()

    def setup_model(self):
        # Define prior
        self.prior = self.define_prior()

        # Define and setup networks
        self.set_summary_network()
        self.set_inference_network()
        self.set_amortized_posterior()

        # Setup trainer
        self.trainer = Trainer(amortizer=self.amortized_posterior, generative_model=None)

    def dataset_info(self):
        print(f"X_train.shape = {self.X_train.shape}")
        print(f"y_train.shape = {self.y_train.shape}")
        print(f"X_test.shape = {self.X_test.shape}")
        print(f"y_test.shape = {self.y_test.shape}")

    def define_prior(self):
        return tfd.JointDistributionNamed({
            'alpha': tfd.Uniform(low=-5., high=5.),
            'mass_min': tfd.Uniform(low=5., high=20.),
            'mass_max': tfd.Uniform(low=30., high=200.)
        })

    # def set_summary_network(self):
    #     self.summary_network = Sequential([
    #         Flatten(input_shape=self.input_shape),
    #         Dense(128, activation='relu'),
    #         BatchNormalization(),
    #         Dense(64, activation='relu'),
    #         BatchNormalization(),
    #         Dense(30),  # Summary dimension
    #     ])

    def set_summary_network(self):
        # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
        # self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
        self.summary_network = Sequential([
            # InputLayer(input_shape=self.input_shape),
            # Dense(2),
            # Dropout(0.5),
            # BatchNormalization(),
            # Reshape((200, 1000)),
            Flatten(input_shape=self.input_shape),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(30),  # Summary dimension
            # DeepSet(30)
            # SequenceNetwork(summary_dim=120, num_conv_layers=5, bidirectional=True)
        ])
        # self.summary_network = InvariantNetwork(12),
        # self.summary_network = SequenceNetwork(summary_dim=120, num_conv_layers=5, bidirectional=True)

    # def set_summary_network(self):
    #     self.summary_network = DeepSet(12)

    def set_inference_network(self):
        self.inference_network = InvertibleNetwork(
            num_params=3,  # Assuming 3 parameters to estimate
            # num_blocks=2,
            # subnet_constructor=self.summary_network  # Utilize summary network in the subnet construction
        )

    def set_amortized_posterior(self):
        self.amortized_posterior = AmortizedPosterior(self.inference_network, self.summary_network)

    def train(self, epochs=100):
        simulations_dict = {'sim_data': self.X_train, 'prior_draws': self.y_train}
        self.trainer.train_offline(simulations_dict, batch_size=self.batch_size, epochs=epochs)

    def predict(self, X=None):
            # X = self.convert_to_tensor([X])[0]
            # X = np.reshape(X, (X.shape[0], -1, X.shape[2]))
        print()
        print(f"expected_means: \n{self.y_test[15]}")
        print()
        print("predicted_means:")
        samples = self.amortized_posterior.sample({'summary_conditions': self.X_test[15]}, n_samples=1000)
        predicted_means = np.mean(samples, axis=1)
        np.set_printoptions(suppress=True)
        print(predicted_means)
        return predicted_means
        # return np.mean(self.amortized_posterior.sample({'summary_conditions': self.X_test[5:15]}, n_samples=1000), axis=1)

    def save_model(self, filepath):
        save_model(self.amortized_posterior.inference_net, filepath + '_inference_net')
        save_model(self.amortized_posterior.summary_net, filepath + '_summary_net')

    def load_model(self, filepath):
        self.inference_network = load_model(filepath + '_inference_net')
        self.summary_network = load_model(filepath + '_summary_net')
        self.set_amortized_posterior()

    def convert_to_tensor(self, data):
        return [tf.convert_to_tensor(d, dtype=tf.float32) for d in data]

    # def plot(self, predictions: np.ndarray, true: np.ndarray, save_folder: str,
    #      filename_prefix: str):
    #     predictions = np.mean(predictions, axis=1)
    #     print(predictions.dtype)
    #     labels = ["alpha", "mass_min", "mass_max"]
    #     idx = predictions.shape[0]
    #
    #     print("y_test shape:", true.shape)
    #
    #     if save_folder[-1] != '/':
    #         save_folder += '/'
    #     # print("starting the plotting loop")
    #     for i in tqdm(range(idx), desc='examples', leave='False'):
    #         path = osp.join(save_folder, f'{filename_prefix}_{i}')
    #         # print("idx:", idx, "and i:", i)
    #         samples = predictions[i]
    #
    #         # print(f"loop no. {i}")
    #
    #         print(samples.shape)
    #
    #         figure = corner.corner(
    #             samples,
    #             labels=labels,
    #             truths=true[i],
    #             quantiles=[0.16, 0.5, 0.84],
    #             show_titles=True,
    #             title_kwargs={"fontsize": 12}
    #         )
    #         # print("done defining figure")
    #
    #         # if i == (idx-1):
    #         #     figure.savefig(path)
    #         plt.close(figure)

