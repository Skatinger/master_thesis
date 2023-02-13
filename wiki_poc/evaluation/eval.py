import re
# import pandas as pd
# from metrics import TopKAccuracy
# from tqdm import tqdm
# Evaluation Methods
# for each example, we need:
# - the name of the entity
# - the predicted tokens
# - the actual tokens (?)


# evaluation helper
class Eval:

    # metrics
    # - correct top 1
    # - correct top 5
    # - correct within all predictions (e.g. if entity is 2x in top 5, it is correct twice out of 5)
    # - correct within page
    def __init__(self, dataset, ground_truth, shard_size=None, shard_index=None):
        # if shard_size given, get specified shard from ground_truth
        if shard_size is not None:
            nb_shards = len(ground_truth) / shard_size
            self.gt = ground_truth.shard(nb_shards=nb_shards, index=shard_index)
        else:
            self.gt = ground_truth
        self.dataset = dataset
        assert (len(self.dataset) == len(self.gt)), "Dataset lengths did not match ({} and {} rows)".format(len(self.dataset), len(self.gt))

        # join datasets together for analysis
        self.joined = self.join_sets(self.dataset, self.gt)

    def evaluate(self):
        self.format_data()
        self.compute_accuracy_matrix()
        results = self.compute_metrics()
        return results

    def join_sets(self, dataset, gt):
        result = gt.add_row(name='predictions', column=dataset['predictions'])
        result = gt.add_row(name='scores', column=dataset['scores'])
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
    def format_data(self):
        # format dataset to predictions and labels
        self.predictions = self.joined['predictions']
        self.labels = self.joined['labels']
        # for every example, we need the same label for every mask
        # so we just repeat the label for each prediction
        self.accurate_labels = []
        for example in self.joined:
            self.accurate_labels.append(example['title'] * len(example['predictions']))

    # computes a matrix of shape (len(dataset), len(predictions))    
    def compute_accuracy_matrix(self):
        # for each example, compute the accuracy for each prediction
        # 1 if correct, 0 if not
        # return a matrix of shape (len(dataset), len(predictions))
        # each row is a example, each column is a prediction
        self.accuracy_matrix = []
        for label, predictions in zip(self.accurate_labels, self.predictions):
            # compute the accuracy mask for this example, should be [0, 1, 0, 1, 0] for example
            self.accuracy_matrix.append(self.match_mask(label, predictions))

    def top_k_accuracy(self, k):
        # compute how many times the correct label was in the top 5 predictions
        total = 0
        for example in self.accuracy_matrix:
            total += min(sum(example[:k], + [1]))
        return total / len(self.accuracy_matrix)

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
    def match_mask(self, entity, predictions):
        regex = self.name_regex(entity)
        res = [int(bool(re.match(regex, pred))) for pred in predictions]
        return res
