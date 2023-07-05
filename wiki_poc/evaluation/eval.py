import re
from math import floor
import logging
from typing import Dict, Union, Optional, List, Any, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
# from metrics import TopKAccuracy
# from tqdm import tqdm
# Evaluation Methods
# for each example, we need:
# - the name of the entity
# - the predicted tokens
# - the actual tokens (?)


# evaluation helper
class Evaluator:
    """_summary_
        # metrics
        # - correct top 1
        # - correct top 5
        # - correct within all predictions (e.g. if entity is 2x in top 5, it is correct twice out of 5)
        # - correct within page
    
    Args:
        dataset: the dataset with predictions, should contain a column named "predictions" and a column named "scores" OR
                 for single prediction datasets, a column named "prediction" and a page_id column
        ground_truth: the ground truth dataset, should contain columns:
                     - "masks" which contains the masked strings
                     - "title" which contains the title of the page representing the entity of a page
                     - "id" which contains the id of the page
                     - "sequence_number" which contains the sequence number of the page, e.g. the number of a chunk in a page
        shard_size: if the dataset was processed in shards, this is the size of the shard (e.g. 1000)
    
    Internals:
        self.joined: dataset with predictions and ground truth joined together after initialization
        self.predictions: Contains all predictions of the passed dataset. Available after format_data() is called.
        self.labels: Represents the exact strings which were masked. Available after format_data() is called.
                     This is used to compare the predictions with the exact ground truth, to evaluate exact precision.
        self.accurate_labels: Represents the title of the page which was masked. Available after format_data() is called.
                     This is used to evaluate the accuracy of the predictions on a per-page basis, as the goal is to
                     predict the correct entity for a page, not the exact masked string. This assumes that all masks
                     for a page represent the same entity. (e.g. if a page has 2 masks, they are both for the same entity,
                     but could differ as in the case of "Obama" and "Barack Obama").


    Returns:
        _type_: _description_
    """

    def __init__(self, dataset: Dataset = None, ground_truth: Dataset = None, shard_size: int = None):
        if dataset is not None and ground_truth is not None:
            self.init(dataset, ground_truth, shard_size)
        else:
            print("No dataset given. Please call init(**args) to initialize the evaluator.")
        pass

    def init(self, dataset: Dataset, ground_truth: Dataset, shard_size: int = None):
        """initializes the evaluator class with the passed datasets if any are given"""
        # if shard_size given, get all examples from the ground_truth with the matching ids from the predictions dataset
        if shard_size is not None:
            logging.info("Filtering ground truth dataset by ids from predictions dataset...")
            self.g_t = ground_truth.filter(lambda x: x['id'] in dataset['page_id'], with_indices=False, num_proc=8)#.sort('id')
        else:
            self.g_t = ground_truth.sort('id')
        self.dataset = dataset.sort('page_id')
        assert (len(self.dataset) == len(self.g_t)), \
            f"Dataset lengths did not match ({len(self.dataset)} and {len(self.g_t)} rows)"

        # join datasets together for analysis
        self.joined = self.join_sets(self.dataset, self.g_t)

        # initialize additional variables
        self.predictions = None
        self.labels = None
        self.accurate_labels = None
        self.accuracy_matrix = None
        self.__format_data()
        self.mask_count = sum([len(x) for x in self.labels])

    def evaluate(self):
        """prepares data for evaluation and computes the metrics

        Returns:
            _type_: dict with metrics and results
        """
        self.compute_accuracy_matrix()
        results = self.compute_metrics()
        return results

    def join_sets(self, dataset: Dataset, g_t: Dataset) -> Dataset:
        """joins the passed datasets together. Assumes they are correctly sorted by id.

        Args:
            dataset (Dataset): predictions dataset
            g_t (Dataset): ground truth dataset

        Returns:
            Dataset: joined dataset with predictions and scores added to ground truth dataset
        """
        result = g_t.add_column(name='predictions', column=dataset['predictions'])
        result = result.add_column(name='scores', column=dataset['scores'])
        return result

    def compute_metrics(self):
        results = {
            'per-example': {
                'top1-accuracy': self.top_k_accuracy(1),
                'top5-accuracy': self.top_k_accuracy(5)
            },
            'per-page': {
                'top1-accuracy': []
            }
        }
        return results

    def join_examples_by_page(self):
        """joins the examples of the joined dataset by page, so that each page is represented by one example.
           This allows to compute metrics on a per-page basis, e.g. top 1 accuracy per page more easily.
        """
        assert len(self.joined) > 0, "Joined dataset not ready or empty"

        # use pandas to group by page id as its faster than using the datasets library
        df = pd.DataFrame(self.joined)

        # group the rows by their page_id, then sort them by their sequence_number
        grouped_df = df.groupby('id')[["id", "title", "texts", "masks", "predictions", "scores"]].aggregate({
            'id': lambda x: x.iloc[0],
            'title': lambda x: x.iloc[0],
            'texts': ' '.join,
            'masks': lambda x: [item for sublist in x for item in sublist],
            'predictions': lambda x: [item for sublist in x for item in sublist],
            'scores': lambda x: [item for sublist in x for item in sublist]
            })

        # reset the index to be sequential again
        grouped_df = grouped_df.reset_index(drop=True)

        # convert the dataframe back to a dataset
        self.by_page = Dataset.from_pandas(grouped_df)

    # formats given dataset to predictions and labels
    def __format_data(self):
        """fills the internal variables predictions and labels with the predictions
           and labels extracted from the joined dataset. Also fills the accurate_labels
           variable with the labels for each prediction.
        """
        # format dataset to predictions and labels for easy access
        self.predictions = self.joined['predictions']
        self.labels = self.joined['masks']
        # for every example, we need the same label for every mask, representing the correct entity for the page
        # to get that, we use the title of the page as the label
        self.accurate_labels = []
        for example in self.joined:
            self.accurate_labels.append([example['title']] * len(example['predictions']))

    # computes a matrix of shape (len(dataset), len(predictions))    
    def compute_accuracy_matrix(self):
        """using the predictions and labels, computes a matrix of shape (len(dataset), len(predictions))
           where each row represents an example and each column represents a masked token for that example.
           each cell contains an array of length 5, which represent the top 5 predictions.
        """
        # for each example, compute the accuracy for each prediction
        # 1 if correct, 0 if not
        # return a matrix of shape (len(dataset), len(predictions))
        # each row is a example, each column is a prediction
        self.accuracy_matrix = [[] for _ in range(len(self.dataset))]
        # iterate over all examples with their predictions, scores and labels
        iterator = zip(self.accurate_labels, self.dataset['predictions'], self.dataset['scores'])
        for index, (labels, predictions, scores) in enumerate(iterator):
            """
            label is a list of labels for each prediction in the example
            predictions is a list of lists of predictions for the example
            scores is a list of lists of scores for the predictions in the example
            compute the accuracy mask for this example, should be [0, 1, 0, 1, 0] for example
            for every label, check all its corresponding predictions and append scores with it"""
            for label, prediction_group, scores_group in zip(labels, predictions, scores):
                # for each prediction, check if it matches the label
                # if it does, append the score to the accuracy mask
                # if it doesn't, append 0
                self.accuracy_matrix[index].append(self.match_mask(label, prediction_group, scores_group))
    

    # def match_mask(self, label: str, predictions: List[str], scores: List[float]) -> List[float]:

    
    def compute_accuracy_matrix_per_page(self):
        """computes the accuracy matrix per page, i.e. for each page, we compute the accuracy
           for each prediction, and then average the results for each page.
        """
        self.join_examples_by_page()
        self.accuracy_matrix_by_page = [[] for _ in range(len(self.by_page))]
        # iterate over all examples with their predictions, scores and labels
        iterator = zip(self.accurate_labels, self.dataset['predictions'], self.dataset['scores'])
        for index, (labels, predictions, scores) in enumerate(iterator):
            """
            label is a list of labels for each prediction in the example
            predictions is a list of lists of predictions for the example
            scores is a list of lists of scores for the predictions in the example
            compute the accuracy mask for this example, should be [0, 1, 0, 1, 0] for example
            for every label, check all its corresponding predictions and append scores with it"""
            for label, prediction_group, scores_group in zip(labels, predictions, scores):
                # for each prediction, check if it matches the label
                # if it does, append the score to the accuracy mask
                # if it doesn't, append 0
                self.accuracy_matrix_by_page[index].append(self.match_mask(label, prediction_group, scores_group))


    def top_k_accuracy_per_mask(self, k):
        """compute how many times the correct label was in the top 5 predictions"""
        total = 0
        for index, example in enumerate(self.accuracy_matrix):
            # in some cases the number of predictions is not 5 but 1, this is a bug
            # we ignore these examples for now
            if len(example) == 5 and len(example[0]) == 1:
                logging.info("Malformed example #{} ({})".format(index, example))
                continue
            for prediction in example:
                # if the correct label is in the top 5 predictions, add 1 to total
                if any([prediction[i] for i in range(k)]):
                    total += 1
        return total / self.mask_count

    def top_k_accuracy_per_page(self, k):
        """computes the number of times the correct label was in the top k predictions for each page,
           e.g. how often was a page correctly predicted in the top 5 predictions for the full page.

        Returns:
            _type_: _description_
        """
        pass

    # builder for a regex which checks if a string approximately matches the given entity
    # same regex as used for masking the wikipedia dataset e.g. only NER predictions fitting this regex
    # were actually masked
    def name_regex(self, entity):
        regex = ".*".join(entity.split(" "))
        # remove any special characters
        regex = re.sub(r'[^a-zA-Z0-9 ]', '', regex)
        regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity.split(" ")])
        return regex

    # takes a entity as String and an array of predictions
    # returns an array of integers indicating for each prediction wether it fit the requirement
    # to match the given entity
    # 1 is true, 0 is false
    def match_mask(self, entity, predictions, scores):
        regex = self.name_regex(entity)
        res = [int(bool(re.match(regex, pred))) * score for pred, score in zip(predictions, scores)]
        return res
    
    def name_match(self, entity, prediction):
        regex = self.name_regex(entity)
        return int(bool(re.match(regex, prediction)))

    def binary_match_mask(self, entity: str, predictions: List[str]) -> List[int]:
        """returns an array of integers indicating for each prediction wether it fit the requirement
           to match the given entity
           1 is true, 0 is false

        Args:
            entity (str): name of the entity, e.g. "Abraham Lincoln"
            predictions (List[str]): list of predictions for the entity
        """
        return [int(bool(re.match(self.name_regex(entity), pred))) for pred in predictions]

    def batched_binary_match_mask(self, entities: List[str], predictions: List[List[str]]) -> List[List[int]]:
        """returns a list of lists of integers indicating for each prediction wether it fit the requirement
           to match the given entity
           1 is true, 0 is false

        Args:
            entities (List[str]): list of entities, e.g. ["Abraham Lincoln", "George Washington"]
            predictions (List[List[str]]): list of predictions for each entity
        """
        return [self.binary_match_mask(entity, prediction) for entity, prediction in zip(entities, predictions)]

    def average_certainty_per_correct_prediction(self):
        """computes the average score over all predictions

        Returns:
            String: description of the metric
            _type_: list of floats
        """
        score = 0
        correct_predictions_count = 0
        for example in self.accuracy_matrix:
            for mask in example:
                score += sum(mask)
                correct_predictions_count = len(np.nonzero(mask))
        return 'avg_score_per_mask', score / correct_predictions_count
    
    # for a given page, return the 
    def ranks_one(self, label, predictions, scores):
        """"""
    
    def accuracy_per_page(self):
        """computes the accuracy for each page, i.e. for each page, we compute the accuracy
           for each prediction, and then average the results for each page.

        Returns:
            String: description of the metric
            List[float]: list of floats
        """
        accuracies = []
        for example in self.accuracy_matrix_by_page:
            accuracies.append(sum(example) / len(example))
        return 'accuracy_per_page', accuracies

    @staticmethod
    def accuracy_per_page(examples: List[Dict]) -> Tuple[str, List[float]]:
    

    def most_frequent_prediction_accuracy(self, examples: List[Dict], top_k: int = 5) -> float:
        """takes a list of examples containing a list of predictions and a label
           and computes the accuracy of the most frequent prediction, e.g. how often
           the number one prediction was the correct one. If multiple predictions are
           passed, the prediction is counted as correct if it is within the top 5 most frequent
           predictions for an example.

        Args:
            examples (List[Dict]): a

        Returns:
            float: the accuracy of the most frequent prediction over all examples
        """
        num_correct = 0
        for example in examples:
            # for models which returned a list of predictions (e.g. fill-mask models), we compute which prediction
            # was the most frequent and check if it matched. predictions is a list of predictions for each mask
            if isinstance(example['predictions'], list):
                # initialize a dictionary which counts how often each prediction was made, with the correct label
                # as the first entry. (e.g. {'Abraham Lincoln': 0, 'George Washington': 2, ...})
                label = example['masks'][0]
                prediction_count = { label: 0 }
                # each prediction can contain several suggestions
                for prediction in example['predictions']:
                    for suggestion in prediction:
                        # skip empty suggestions
                        if suggestion == '':
                            continue
                        # if the suggestion fullfills the name_match requirement, add it to the prediction_count as correct
                        if self.name_match(label, suggestion):
                            prediction_count[label] += 1
                        # if it does not match, add it to counter as exact prediction
                        elif suggestion not in prediction_count.keys():
                            prediction_count[suggestion] = 1
                        else:
                            prediction_count[suggestion] += 1                    
                # check if the the label was within the top k predictions
                top_k_predictions = sorted(prediction_count.items(), key=lambda x: x[1], reverse=True)[:top_k]
                was_in_top_k = example['masks'] in [label for (label, _) in top_k_predictions]
                num_correct += int(was_in_top_k)
            
            # for models which returned a single prediction instead of a list of predictions
            # we can only check if the prediction matched
            else:
                num_correct += self.name_match(example['masks'], example['predictions'])


        accuracy = num_correct / len(examples)

        return accuracy

