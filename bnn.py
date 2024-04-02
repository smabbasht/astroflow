import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import keras
from keras import Model, Sequential 
from keras.layers import InputLayer, Dense
from tensorflow.keras.initializers import GlorotNormal

tfd = tfp.distributions
tfpl = tfp.layers

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
EVENT_SHAPE = 3


class BayesianNN(tf.keras.Model):
    
    def __init__(self, input_shape, input_data):
            super(BayesianNN, self).__init__()
            self.model_input_shape = input_shape
            self.model = self.build_bayesian_nn()
            self.X_train = tf.convert_to_tensor(input_data[0], dtype=tf.float32)
            self.y_train = tf.convert_to_tensor(input_data[2], dtype=tf.float32)
            self.X_test = tf.convert_to_tensor(input_data[1], dtype=tf.float32)
            self.y_test = tf.convert_to_tensor(input_data[3], dtype=tf.float32)

    
    def build_bayesian_nn(self):
        model = Sequential([
            InputLayer(input_shape=self.model_input_shape),
            tfpl.DenseFlipout(units=128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tfpl.DenseFlipout(units=64, activation='relu'),
            tfpl.DenseFlipout(units=16, activation='relu'),
            tf.keras.layers.Flatten(),
            Dense(tfpl.IndependentNormal.params_size(EVENT_SHAPE)),
            tfpl.IndependentNormal(EVENT_SHAPE, convert_to_tensor_fn=tfp.distributions.Distribution.sample),
            # tfpl.DenseFlipout(units=tfpl.IndependentNormal.params_size(EVENT_SHAPE)),  # Output layer
            # tfpl.IndependentNormal(EVENT_SHAPE, convert_to_tensor_fn=tfp.distributions.Distribution.mean),
            # tfpl.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1), convert_to_tensor_fn=tfp.distributions.Distribution.mean),
            ])

        for layer in model.layers:
            print(layer.name, layer.output_shape)
        return model

    def loss(self, y_true, y_pred):
        # print("y_pred", y_pred)
        # print("y_true", y_true)
        loss1 = -y_pred.log_prob(y_true)
        # kl_loss = sum(model.losses)
        # loss1 = loss1 + kl_loss
        # print("loss", loss1)
        return loss1

    # def build_bayesian_nn(self):
    #     model = Sequential([
    #         InputLayer(input_shape=self.model_input_shape),
    #         tfpl.DenseFlipout(units=128, activation='relu'),
    #         tfpl.DenseFlipout(units=64, activation='relu'),
    #         tfpl.DenseFlipout(units=3),  # Output layer with shape (batch_size, 3)
    #         tfpl.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
    #     ])
    #     return model
    # def __init__(self, input_shape, input_data):
    #     super(BayesianNN, self).__init__()
    #     self.input_shape = input_shape
    #     self.model = self.build_bayesian_nn()
    #     self.X_train = tf.convert_to_tensor(input_data[0], dtype=tf.float32)
    #     self.y_train = tf.convert_to_tensor(input_data[2], dtype=tf.float32)
    #     self.X_test = tf.convert_to_tensor(input_data[1], dtype=tf.float32)
    #     self.y_test = tf.convert_to_tensor(input_data[3], dtype=tf.float32)
    #
    #
    # def build_bayesian_nn(self):
    #     inputs = tf.keras.Input(shape=self.input_shape)
    #     x = tfpl.DenseFlipout(units=128, activation='relu')(inputs)
    #     x = tfpl.DenseFlipout(units=64, activation='relu')(x)
    #     x = tfpl.DenseFlipout(units=3)(x)
    #     outputs = tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1))(x)
    #     
    #     model = Model(inputs=inputs, outputs=outputs)
    #     return model

    # def build_bayesian_nn(self):
    #     model = keras.Sequential([
    #         InputLayer(input_shape=self.input_shape),
    #         tfpl.DenseFlipout(units=128, activation='relu'),
    #         tfpl.DenseFlipout(units=64, activation='relu'),
    #         tfpl.DenseFlipout(units=3),  # Output layer with shape (batch_size, 3)
    #         tfpl.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
    #     ])
    #     return model

    def call(self, inputs):
        return self.model(inputs),                                                                
    def save_model(self, path: str):
        self.model.save(path)
        print(f"Model has been saved at '{path}'")

    def load_model(self, path: str):
        self.model.load_weights(path)
        print(f"Model has been loaded from '{path}'")

    def train(self, n_epochs: int, batch_size: int):
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1), loss=self.loss)
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, epochs=n_epochs, batch_size=batch_size)

    def predict(self, input_data):
        predictions = self.model(input_data)
        return predictions

    def eval(self):
        samples = self.model(self.X_test[-2:-1]).sample(1000)
        print(samples)
        return samples

    def plot(self, predictions: np.ndarray, true: np.ndarray, save_folder: str,
         filename_prefix: str):
        predictions = np.mean(predictions, axis=1)
        print(predictions.dtype)
        labels = ["alpha", "mass_min", "mass_max", "sigma_ecc"]
        idx = predictions.shape[0]

        print("y_test shape:", true.shape)

        if save_folder[-1] != '/':
            save_folder += '/'
        # print("starting the plotting loop")
        for i in tqdm(range(idx), desc='examples', leave='False'):
            path = osp.join(save_folder, f'{filename_prefix}_{i}')
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

            if i == (idx-1):
                figure.savefig(path)
            plt.close(figure)




# def build_bayesian_nn(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=input_shape),
#         tfpl.DenseFlipout(units=128, activation='relu'),
#         tfpl.DenseFlipout(units=64, activation='relu'),
#         tfpl.DenseFlipout(units=3),  # Output layer with shape (batch_size, 3)
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),])
#     return model
#
# # Create input distributions
# m1_distribution = tfd.Normal(loc=0.0, scale=1.0)
# m2_distribution = tfd.Normal(loc=0.0, scale=1.0)
#
# # Sample from input distributions to create input data
# m1_samples = m1_distribution.sample((batch_size, 600))
# m2_samples = m2_distribution.sample((batch_size, 600))
#
# # Concatenate m1_samples and m2_samples to create input tensor
# input_data = tf.concat([m1_samples, m2_samples], axis=1)
#
# # Instantiate the model
# model = build_bayesian_nn((input_data.shape[1],))
#
# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))
#
# # Train the model
# model.fit(input_data, y_train, epochs=10, batch_size=32)
#
# # Perform inference and obtain samples
# predictions = model(input_data)  # Assuming input_data is your test data
# samples = predictions.sample(1000)
