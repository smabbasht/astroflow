from sklearn.model_selection import train_test_split
from rich.progress import Progress
import pandas as pd
import numpy as np
import os

LINES_TO_READ = 1000

class Dataset:
    def __init__(self, data_path: str, internal_data_path: str, test_split: int) -> None:
        self.data_path = data_path
        self.test_split = test_split
        self.internal_data_path = internal_data_path
        self.features = []
        self.labels = []


    def preprocess_amanda(self, array_path: str):

        example_folders = os.listdir(self.data_path)

        with Progress() as progress:
            task1 = progress.add_task("[red]Preprocessing Dataaset...", total=len(example_folders))

            for i, example in enumerate(example_folders):
                feature_path = os.path.join(self.data_path, example, "mock_data")
                label_path = os.path.join(self.data_path, example, "example_mock_PE.csv")

                event_files = [file for file in os.listdir(
                    feature_path) if file.startswith('0_')]

                feature3d = []
                for ind, file in enumerate(event_files):
                    table = pd.read_csv(os.path.join(feature_path, file), header=None, usecols=[0, 1, 3], skiprows=[0])
                    # table = pd.csv(os.path.join(dat_files_path, file), sep=" ", skiprows=[0], header=True)
                    feature3d.append(table)

                feature3d = np.stack(feature3d)
                # print(feature3d.shape, i)

                self.features.append(feature3d)
                # self.features[i] = feature3d

                # populating labels
                # label = pd.read_csv(label_path, skiprows=[0], header=None)
                label = pd.read_csv(label_path, skiprows=1, header=None)
                self.labels.append(np.asarray(
                    label.iloc[0].values.tolist()))
                # print(label)


                progress.update(task1, advance=1)

        self.features = np.asarray(self.features)
        self.labels = np.asarray(self.labels)

        print('type of features: ', type(self.features))
        print('shape of features:', self.features.shape)
        print('type of labels: ', type(self.labels))
        print('shape of labels:', self.labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=self.test_split,
                                                            random_state=42,
                                                            shuffle=True)
        print("y_test shape received after sklearn split", y_test.shape)
        os.makedirs(array_path, exist_ok=True)
        np.save(os.path.join(array_path, "X_train.npy"), X_train)
        np.save(os.path.join(array_path, "X_test.npy"), X_test)
        np.save(os.path.join(array_path, "y_train.npy"), y_train)
        np.save(os.path.join(array_path, "y_test.npy"), y_test)

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, array_path):
        example_folders = os.listdir(self.data_path)

        with Progress() as progress:
            task1 = progress.add_task("[red]Preprocessing Dataset...", total=len(example_folders))
            for i, example in enumerate(example_folders):
                outer_break=False
                example_path = os.path.join(self.data_path, example)

                # populating features
                dat_files_path = os.path.join(example_path, self.internal_data_path, 'posteriors')
                dat_files = [file for file in os.listdir(
                    dat_files_path) if file.endswith('.dat')]

                feature3d = []
                for ind, file in enumerate(dat_files):
                    skip = []
                    with open(os.path.join(dat_files_path, file), 'r') as filecount:
                        total_lines = sum(1 for line in filecount) - 1  # Exclude the first line
                    if total_lines >= LINES_TO_READ:
                        skip = sorted(np.random.choice(np.arange(total_lines), size=(total_lines - LINES_TO_READ), replace=False))
                        print(f"Processing file {file} for event {example} with {total_lines} samples and filtered samples {total_lines-len(skip)} for {ind+1}th time.")
                    else:
                        print(f"Processing file {file} for event {example} with {total_lines} samples.")
                        outer_break=True
                        break
                        # raise AssertionError("The .dat files for events contains fewer than the minimum samples i-e 500 which is the minimum required for training the model. Please check the data.")
                    skip=list(map(lambda x: x+1, skip))
                    table = pd.read_table(os.path.join(dat_files_path, file), sep=" ", skiprows=[0]+skip, header=None)
                    feature3d.append(table)



                try:
                    feature3d = np.stack(feature3d)
                except Exception as e:
                    print(f"\n\n\nException {e} occured so I skipped this example.\n\n\n")
                    continue
                print(feature3d.shape, i)

                if outer_break==True:
                    continue

                self.features.append(feature3d)
                # self.features[i] = feature3d

                # populating labels
                label_path = os.path.join(example_path, self.internal_data_path, 'configuration.dat')
                data = pd.read_table(label_path, comment='#', header=None)
                print(data)
                self.labels.append(np.asarray(
                    data.iloc[0].values.tolist()))


                progress.update(task1, advance=1)

        self.features = np.asarray(self.features)
        self.labels = np.asarray(self.labels)
        print('type of features: ', type(self.features))
        print('shape of features:', self.features.shape)
        print('type of labels: ', type(self.labels))
        print('shape of labels:', self.labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=self.test_split,
                                                            random_state=42,
                                                            shuffle=False)
        print("y_test shape received after sklearn split", y_test.shape)
        os.makedirs(array_path, exist_ok=True)
        np.save(os.path.join(array_path, "X_train.npy"), X_train)
        np.save(os.path.join(array_path, "X_test.npy"), X_test)
        np.save(os.path.join(array_path, "y_train.npy"), y_train)
        np.save(os.path.join(array_path, "y_test.npy"), y_test)
        # np.save('dataset_arrays/X_train.npy', X_train)
        # np.save('dataset_arrays/y_train.npy', y_train)
        # np.save('dataset_arrays/X_test.npy', X_test)
        # np.save('dataset_arrays/y_test.npy', y_test)
        return [X_train, X_test, y_train, y_test]

    def combine_arrays(self, arrays_base_paths: list, save=True, save_path=None):
        combined_data = {
            "X_train": [],
            "X_test": [],
            "y_train": [],
            "y_test": []
        }

        for base_path in arrays_base_paths:
            arrays = self.load_data(base_path)
            combined_data["X_train"].append(arrays[0])
            combined_data["X_test"].append(arrays[1])
            combined_data["y_train"].append(arrays[2])
            combined_data["y_test"].append(arrays[3])

        X_train = np.concatenate(combined_data["X_train"], axis=0)
        X_test = np.concatenate(combined_data["X_test"], axis=0)
        y_train = np.concatenate(combined_data["y_train"], axis=0)
        y_test = np.concatenate(combined_data["y_test"], axis=0)

        if save and save_path:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "X_train.npy"), X_train)
            np.save(os.path.join(save_path, "X_test.npy"), X_test)
            np.save(os.path.join(save_path, "y_train.npy"), y_train)
            np.save(os.path.join(save_path, "y_test.npy"), y_test)

        return X_train, X_test, y_train, y_test



    def load_data(self, array_path):
        arrays = [
                np.load(os.path.join(array_path, "X_train.npy")),
                np.load(os.path.join(array_path, "X_test.npy")),
                np.load(os.path.join(array_path, "y_train.npy")),
                np.load(os.path.join(array_path, "y_test.npy")),
        ]
        return arrays

if __name__ == '__main__':
    dataset = Dataset('../blackligo-data-genie/data/', 'realization_0', 0.1)
    dataset.preprocess_data("dataset_arrays_3")
    dataset.combine_arrays(["dataset_arrays_3", "dataset_arrays_2"], save=True, save_path="dataset_arrays")

