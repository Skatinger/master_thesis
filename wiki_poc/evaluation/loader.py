
# run from wiki_poc with `python -m evalution.loader`

import os
import re
import logging
from models.model_runner import get_all_model_names
from datasets import load_dataset


class ResultLoader():

    def result_file_names(self, key):
        return os.listdir(f"results/{key}")

    def load(self, key, model_name = None, model_class = None):
        """loads and returns all results as dict. can be filtered by model_class or model_name
           returns: {"model-name": {"..., data: Dataset"}} 
        """
        file_names = self.result_file_names(key)
        if model_name:
            return self.load_by_name(key, model_name)
        if model_class:
            return self.load_by_class(key, model_class)

        models = {}
        for file_name in file_names:
            name, size, config, inputsize = self.parse_name(file_name)
            logging.info(f"loading dataset from: results/{key}/{file_name}")

            if not models.get(name):
                models[name] = {}
            
            models[name][size] = {
                "size": float(re.sub('[mb]', '.', size)), # store size as float
                "inputsize": inputsize,
                config: load_dataset("json", data_files=f"results/{key}/{file_name}")
            }
        return models
    
    def load_by_class(self, model_class):
        pass

    def load_by_name(self, model_name):
        pass

    def parse_name(self, filename):
        model_name, rest = filename.split("-")  # Split the filename by underscores
        size, config, inputsize = rest.split("_")  # Split the remaining part by hyphen to get size, config, and inputsize
        inputsize = inputsize.rstrip(".json")  # Remove the ".json" extension from inputsize
        # size = # float(re.sub('[mb]', '.', size))
        return model_name, size, config, inputsize


def main():
    loader = ResultLoader()
    dict = loader.load("1000_basic")
    print(dict)
    

if __name__ == "__main__":
    main()
