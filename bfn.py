from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

from tensorflow.keras.models import save_model, load_model
from keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.layers import InputLayer

from bayesflow.trainers import Trainer
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import DeepSet, SequenceNetwork, InvariantNetwork

import os
import tqdm
import corner
import numpy as np
import matplotlib.pyplot as plt 

class BayesianFlowNetwork:
    def __init__(self, input_shape, input_data, batch_size):
        self.input_shape = input_shape
        self.prior = self.define_prior()
        self.trainer=None 
        self.summary_network=None 
        self.inference_network=None
        self.amortized_posterior=None
        self.batch_size = batch_size
        self.X_train, self.X_test = self.convert_to_tensor(input_data[:2])
        self.y_train, self.y_test = self.convert_to_tensor(input_data[2:])

        self.setup_model()

    def setup_model(self):
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
    # #     # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
    # #     # self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
    #     self.summary_network = Sequential([
    #         # SequenceNetwork(summary_dim=128, num_conv_layers=5, bidirectional=True),
    #         Flatten(input_shape=self.input_shape),
    #         Dense(128, activation='relu'),
    #         BatchNormalization(),
    #         Dense(64, activation='relu'),
    #         BatchNormalization(),
    #         Dense(30),  # Summary dimension
    #     ])

    # def set_summary_network(self):
    #     self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
    #     self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
        # self.summary_network = Sequential([
        #     # SequenceNetwork(summary_dim=128, num_conv_layers=5, bidirectional=True),
        #     Flatten(input_shape=self.input_shape),
        #     Dense(128, activation='relu'),
        #     BatchNormalization(),
        #     Dense(64, activation='relu'),
        #     BatchNormalization(),
        #     Dense(32)
        #     # Reshape((32, 2)),
        #     # SequenceNetwork(summary_dim=32, num_conv_layers=5, bidirectional=True),
        # ])
        # self.summary_network = SequenceNetwork(summary_dim=32, num_conv_layers=5, bidirectional=True),
        # print("Summary Network Defined")

    # def set_summary_network(self):
    #     self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
    #     self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
    #     # self.summary_network = Sequential([
    #     #     # InputLayer(input_shape=self.input_shape),
    #     #     # Dropout(0.5),
    #     #     # BatchNormalization(),
    #     #     # Reshape((200, 1000)),
    #     #     Flatten(input_shape=self.input_shape),
    #     #     # Flatten(),
    #     #     Dense(512, activation='relu'),
    #     #     # BatchNormalization(),
    #     #     Dense(128, activation='relu'),
    #     #     BatchNormalization(),
    #     #     Dense(64, activation='relu'),
    #     #     # BatchNormalization(),
    #     #     # Dense(2),
    #     #     # Reshape((200, 1000)),
    #     #
    #     #     Dense(30),  # Summary dimension
    #     #     # DeepSet(30)
    #     #     # SequenceNetwork(summary_dim=120, num_conv_layers=10, bidirectional=True)
    #     # ])
    #     # self.summary_network = InvariantNetwork(12),
    #     self.summary_network = Sequential([
    #         SequenceNetwork(summary_dim=512, num_conv_layers=5, bidirectional=True),
    #         Reshape((256, 2)),
    #         DeepSet(128),
    #         Reshape((64, 2)),
    #         SequenceNetwork(summary_dim=32, num_conv_layers=5, bidirectional=True),
    #         ])

    #     self.summary_network = Sequential([
    #         Flatten(input_shape=self.input_shape),
    #         Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    #         BatchNormalization(),
    #         Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    #         BatchNormalization(),
    #         Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    #         ])

    # def set_summary_network(self):
    #     self.summary_network = Sequential([
    #     Flatten(input_shape=self.input_shape),
    #     Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    #     BatchNormalization(),
    #     Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    #     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    #     Dropout(0.5),
    #     BatchNormalization(),
    #     Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    #     Dropout(0.5),
    #     BatchNormalization(),
    #     Dense(30),  # Summary dimension
    # ])
        # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, self.X_train.shape[2]))
        # self.X_test  = np.reshape(self.X_test,  (self.X_test.shape[0],  -1, self.X_test.shape[2]))
        # self.summary_network = DeepSet(12)

    def set_summary_network(self):

        self.summary_network = Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256),
        Reshape((128, 2)),

        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),

        DeepSet(30)
    ])

    def set_inference_network(self):
        self.inference_network = InvertibleNetwork(
            num_params=8,  # parameters to estimate
            num_coupling_layers=8,
            # coupling_design="spline",
            use_act_norm=True,
            use_soft_flow=True,
            # coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False},
        )
        print("Inference Network Defined")

    def set_amortized_posterior(self):
        self.amortized_posterior = AmortizedPosterior(self.inference_network, self.summary_network)
        print("Amortized Posterior Defined")

    def train(self, epochs=100):
        simulations_dict = {'sim_data': self.X_train, 'prior_draws': self.y_train}
        self.trainer.train_offline(simulations_dict, batch_size=self.batch_size, epochs=epochs, 
                                   optimizer=Adam(learning_rate=0.0005), checkpoint_path="models")
        # self.trainer.train_offline(simulations_dict, batch_size=self.batch_size, epochs=epochs, optimizer=RMSprop(learning_rate=0.001, rho=0.9))

    def predict(self, X=None):
            # X = self.convert_to_tensor([X])[0]
            # X = np.reshape(X, (X.shape[0], -1, X.shape[2]))
        print()
        np.set_printoptions(suppress=True)
        # print(f"expected_means: \n{self.y_test[:15]}")
        # print()
        # print("predicted_means:")
        samples = self.amortized_posterior.sample({'summary_conditions': self.X_test}, n_samples=5000)
        # predicted_means = np.mean(samples, axis=1)
        # np.set_printoptions(suppress=True)
        # print(predicted_means)
        return samples
        # return np.mean(self.amortized_posterior.sample({'summary_conditions': self.X_test[5:15]}, n_samples=1000), axis=1)

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

    def normalize_dataset(self, dim=3):
        scaler = StandardScaler()

        # Reshape to 2D (samples, features)
        original_shape_train = self.X_train.shape  # Keep the original shape
        original_shape_test = self.X_test.shape

        # Reshape for training data
        X_train_reshaped = tf.reshape(self.X_train, (-1, original_shape_train[-2]*original_shape_train[-1]))
        X_test_reshaped = tf.reshape(self.X_test, (-1, original_shape_test[-2]*original_shape_test[-1]))

        # Fit on training data only
        scaler.fit(X_train_reshaped)

        # Transform both training and testing data
        X_train_scaled = scaler.transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        # Reshape back to original
        self.X_train = tf.reshape(X_train_scaled, original_shape_train)
        self.X_test = tf.reshape(X_test_scaled, original_shape_test)

    # def normalize_dataset(self, dim=3):
    #     scaler = StandardScaler()
    #     if dim==2:
    #         self.X_train = self.X_train.reshape(-1, self.X_train.shape[-1])  # Reshape to 2D
    #         self.X_test = self.X_test.reshape(-1, self.X_test.shape[-1])     # Reshape to 2D
    #     scaler.fit(self.X_train)
    #     self.X_train = scaler.transform(self.X_train).reshape(self.X_train.shape)
    #     self.X_test = scaler.transform(self.X_test).reshape(self.X_test.shape)

    def plot(self, predictions: np.ndarray, true: np.ndarray, save_folder: str,
         filename_prefix: str):
        labels = ["alpha_1","alpha_2","mass_min","mass_max","break_fraction","delta_m","beta","lamb"]

        idx = predictions.shape[0]

        print("y_test shape:", true.shape)

        if save_folder[-1] != '/':
            save_folder += '/'
        # print("starting the plotting loop")
        for i in tqdm.tqdm(range(idx), desc='examples', leave='False'):
            path = os.path.join(save_folder, f'{filename_prefix}_{i}')
            # print("idx:", idx, "and i:", i)
            samples = predictions[i]

            # print(f"loop no. {i}")

            print(samples.shape)

            figure = corner.corner(
                samples,
                labels=labels,
                truths=true[i],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )
            # print("done defining figure")

            # if i == (idx-1):
            figure.savefig(path)
            plt.close(figure)

