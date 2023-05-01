


class Metric:

    """
    Base class for all metrics.

    """

    def __init__(self):
        self.acc_matrix = []
        self.accurate_labels = []
        self.predictions = []

    # @staticmethod
    # def __call__(*args, **kwargs):
    #     raise NotImplementedError

    @staticmethod
    def compute(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def aggregate(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def name(*args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()

    def match_mask(self, label, predictions):
        raise NotImplementedError

    # computes a matrix of shape (len(dataset), len(predictions))
    def accuracy_matrix(self):
        if len(self.acc_matrix) > 0:
            return self.acc_matrix
        # for each example, compute the accuracy for each prediction
        # 1 if correct, 0 if not
        # return a matrix of shape (len(dataset), len(predictions))
        # each row is a example, each column is a prediction
        for label, predictions in zip(self.accurate_labels, self.predictions):
            # compute the accuracy mask for this example, should be [0, 1, 0, 1, 0] for example
            self.acc_matrix.append(self.match_mask(label, predictions))


class TopKAccuracy(Metric):

    """
    Computes the top-k accuracy for a given set of predictions and labels.

    """

    def __init__(self, k=1):
        self.k = k
        super().__init__()

    def __call__(self, predictions, labels):
        return self.compute(predictions, labels)

    def compute(self, predictions, labels):
        return self.aggregate(predictions, labels)

    def aggregate(self, predictions, labels):
        return (predictions == labels).sum() / len(labels)

    def name(self):
        return "Top{}Accuracy".format(self.k)
    
    def match_mask(self, label, predictions):
        raise NotImplementedError

class OverallPrecision(Metric):

    """
    Computes the overall precision for a given set of predictions and labels.

    """

    def __call__(self, predictions, labels):
        return self.compute(predictions, labels)

    def compute(self, predictions, labels):
        return self.aggregate(predictions, labels)

    def aggregate(self, predictions, labels):
        return (predictions == labels).sum() / len(labels)

    def name(self):
        return "OverallPrecision"
