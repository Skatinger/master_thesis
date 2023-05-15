import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

print(matplotlib.get_backend())

from evaluation.loader import ResultLoader
from evaluation.single_prediction_eval import SinglePredictionEvaluator

class Plotter():

    @staticmethod
    def plotters():
        return {
            "accuracy-overview": AccuracyOverviewPlotter(),
        }

    @staticmethod
    def parse_options():
        parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
        parser.add_argument("-n", "--name", help="Name of specific chart that should be created", type=str)
        parser.add_argument("-k", "--key", help="Name of the results key", type=str)
        args = parser.parse_args()
        return args.name, args.key

    def plot(self, data, key = "", name = None):
        if name is None:
            self.plot_all(data, key)
        else:
            self.plot_one(data, name, key)
    
    def plot_all(self, data, key):
        for name in self.plotters().keys():
            self.plot_one(data, name, key)

    def plot_one(self, data, name, key):
        self.plotters()[name].build(data, key)


class AccuracyOverviewPlotter(Plotter):

    def build(self, data, key):
        sizes = []
        accuracies = []
        labels = []

        for model, configs in data.items():
            for config, details in configs.items():
                for dataset, results in details.items():
                    if isinstance(results, dict):  # check if result is a dictionary
                        size = details['size']
                        accuracy = results['result']['accuracy']
                        
                        sizes.append(size)
                        accuracies.append(accuracy)
                        labels.append(f"{model}_{config}_{dataset}")

        plt.figure(figsize=(30, 18))
        plt.scatter(sizes, accuracies)

        for i, label in enumerate(labels):
            plt.annotate(label, (sizes[i], accuracies[i]))

        plt.xlabel('Model Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Model Size and Configuration')
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_{key}.png")


def main():
    # key for result dataset from command line arguments
    name, key = Plotter.parse_options()
    print("creating loader")
    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    results = loader.load(key) #, 'bloomz-1b7')


    computed = SinglePredictionEvaluator.compute_metrics(gt, results)

    # import pdb; pdb.set_trace()
    plotter = Plotter()
    plotter.plot(name=name, data=computed, key=key)


if __name__ == "__main__":
    main()