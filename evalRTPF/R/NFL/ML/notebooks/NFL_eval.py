import process_data
from abc import ABC, abstractmethod
import pickle

class NFLEval(ABC):
    def __init__(self, data_dir, features):
        self.data_dir = data_dir
        self.training_data = None
        self.models = {}
        self.features = features
        self.feature_data = None
        self.X_tests = None
        self.feature_test_data = None
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def train(self, model, *args, **kwargs):
        pass
    @abstractmethod
    def evaluate(self):
       pass

    @abstractmethod
    def save_model(self, model_name):
        pass

    @abstractmethod
    def plot(self):
        pass

class NFLMultipleModelsEval(NFLEval):
    def __init__(self, data_dir, features):
        super().__init__(data_dir, features)
    def process(self, test_folders):
        self.training_data = process_data.load_training_data(self.data_dir, test_folders)
        self.feature_data = process_data.feature_selection(self.training_data, self.features)
        print("Done processing")
    def train(self, model, *args, **kwargs):
        self.models = process_data.setup_models(self.features_data, model, *args, **kwargs)
        print("Done training")
    def evaluate(self, test_folders):
        test_folders = process_data.load_test_data(self.data_dir, test_folders)
        self.features_test_data = {key: process_data.test_feature_selection(test_data, self.features) for key, test_data in zip(test_folders.keys(), test_folders.values())}
        X_tests = [self.features_test_data[year] for year in self.features_test_data]
        merged = {}
        for d in X_tests:
            merged.update(d)
        X_tests = merged
        new_X_tests = {}
        for file in X_tests:
            timestep = 0
            for i in range(len(X_tests[file])):
                if timestep not in new_X_tests:
                    new_X_tests[timestep] = [X_tests[file][i]]
                else:
                    new_X_tests[timestep] += [X_tests[file][i]]
                timestep += 0.005
        self.X_tests = new_X_tests
        self.plot()
    def save_model(self, model_name):
        pickle.dump(self.models, open(model_name + ".sav", 'wb'))

    def write_predictions(self, phat_b):
        process_data.write_predictions(self.models, self.features_test_data, self.data_dir, phat_b)

    def plot(self, title=""):
        process_data.plot(self.models, self.X_tests, title)