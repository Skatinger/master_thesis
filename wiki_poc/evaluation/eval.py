import re
from math import floor
import logging
import pandas as pd
from datasets import Dataset
# from metrics import TopKAccuracy
# from tqdm import tqdm
# Evaluation Methods
# for each example, we need:
# - the name of the entity
# - the predicted tokens
# - the actual tokens (?)


# evaluation helper
class Eval:
    """_summary_
        # metrics
        # - correct top 1
        # - correct top 5
        # - correct within all predictions (e.g. if entity is 2x in top 5, it is correct twice out of 5)
        # - correct within page
    
    Args:
        dataset: the dataset with predictions, should contain a column named "predictions" and a column named "scores"
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

    # add data types for parameters
    def __init__(self, dataset: Dataset, ground_truth: Dataset, shard_size: int = None):
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
        assert len(self.joined) > 0, "Joined dataset not ready or empty"

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

    # def top_k_accuracy(self, k):
    #     # compute how many times the correct label was in the top 5 predictions
    #     total = 0
    #     for example in self.accuracy_matrix:
    #         total += min(sum(example[:k], + [1]))
        # return total / len(self.    )

    # builder for a regex which checks if a string approximately matches the given entity
    # same regex as used for masking the wikipedia dataset e.g. only NER predictions fitting this regex
    # were actually masked
    def name_regex(self, entity):
        regex = ".*".join(entity.split(" "))
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
