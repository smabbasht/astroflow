from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


class Dataset:
    def __init__(self, data_path: str, test_split: int, n_examples: int, offset: int) -> None:
        self.data_path = data_path
        self.test_split = test_split
        self.n_examples = n_examples
        # self.n_examples = len(os.listdir(self.data_path))*(n_examples == 0) + (n_examples)
        self.offset = offset
        # self.features = np.zeros(self.n_examples)
        # self.labels = np.zeros(self.n_examples)
        self.features = []
        self.labels = []
        # self.data = self.fetch_data()
        # self.X_train, self.Y_train = self.data[0]
        # self.X_test, self.Y_test = self.data[1]

    def fetch_data(self):
        example_folders = os.listdir(self.data_path)[
            self.offset:(self.n_examples+self.offset)]

        for i, example in enumerate(example_folders):
            example_path = os.path.join(self.data_path, example)

            # populating features
            mock_files = [file for file in os.listdir(
                example_path) if file.endswith('.dat')]
            feature3d = np.asarray([pd.read_table(os.path.join(
                example_path, file), sep=" ", skiprows=1, header=None) for file in mock_files])
            feature3d = np.stack(feature3d)
            print(feature3d.shape, i)
            self.features.append(feature3d)
            # self.features[i] = feature3d

            # populating labels
            file_path = os.path.join(example_path, 'configuration.csv')
            data = pd.read_csv(file_path, usecols=[
                               "alpha", "mass_min", "mass_max", "sigma_ecc"])
            # self.labels[i] = np.asarray(data.iloc[0].values.tolist())
            self.labels.append(np.asarray(
                data.iloc[0].values.tolist()))

        self.features = np.asarray(self.features)
        self.labels = np.asarray(self.labels)
        print('type of features: ', type(self.features))
        print('shape of features:', self.features.shape)
        print('type of labels: ', type(self.labels))
        print('shape of labels:', self.labels.shape)
        x_train, x_test = self.features[int(len(
            self.features)*self.test_split):], self.features[:int(len(self.features)*self.test_split)]
        y_train, y_test = self.labels[int(len(
            self.features)*self.test_split):], self.labels[:int(len(self.labels)*self.test_split)]
        # x_train, x_test, y_train, y_test = train_test_split(self.features,
        #                                                     self.labels,
        #                                                     test_size=self.test_split,
        #                                                     random_state=42,
        #                                                     shuffle=False)
        print("y_test shape received after sklearn split", y_test.shape)
        return [x_train, x_test, y_train, y_test]

    def load_data(self):
        features, labels = self.fetch_data()
        labels = np.array(list(labels.values()))
        features = np.array(list(features.values()))
        X_train, X_test, Y_train, Y_test = train_test_split(features,
                                                            labels,
                                                            test_size=self.test_split,
                                                            random_state=42,
                                                            shuffle=True)
        return [X_train, X_test, Y_train, Y_test]


# dataset = Dataset('../../dataset', 0.2)
