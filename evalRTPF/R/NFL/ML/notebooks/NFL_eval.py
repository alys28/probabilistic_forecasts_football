import process_data
from abc import ABC, abstractmethod


class NFLEval(ABC):
    def __init__(self, data_dir, model_type, model_params, eval_params):
        pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def train(self):
        data = process_data.NFLData(self.data_dir)
        data.load_data()
        data.process_data()
        data.split_data()
        data.save_data()

        model = process_data.NFLModel(self.model_dir, self.model_name, self.model_type, self.model_params)
        model.build_model()
        model.train_model(data.train_data, data.train_labels)
        model.save_model()
    @abstractmethod
    def evaluate(self):
       pass

class NFLMultipleModelsEval(NFLEval):
    def __init__(self, data_dir, model_type, model_params, eval_params):
        super().__init__(data_dir, model_type, model_params, eval_params)

    def process_data(self):
        pass
    
    def train(self):
        pass

    def evaluate(self):
        pass