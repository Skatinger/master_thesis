import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
matplotlib.use('agg')

print(matplotlib.get_backend())

from evaluation.loader import ResultLoader
from evaluation.single_prediction_eval import SinglePredictionEvaluator

class Plotter():

    def __init__(self, key: str = "") -> None:
        self.key = key

    @staticmethod
    def plotters() -> dict:
        return {
            "accuracy-overview": AccuracyOverviewPlotter(),
            "levensthein-distance": LevenstheinDistancePlotter()
        }

    @staticmethod
    def parse_options() -> tuple:
        parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
        parser.add_argument("-n", "--name", help="Name of specific chart that should be created", type=str)
        parser.add_argument("-k", "--key", help="Name of the results key", type=str)
        parser.add_argument("-m", "--model", help="Name of the model that should be plotted", type=str)
        args = parser.parse_args()
        return args.name, args.key, args.model

    def plot(self, data: dict, key: str = "", name = None) -> None:
        self.key = key
        if name is None:
            self.plot_all(data, key)
        else:
            self.plot_one(data, name, key)
    
    def plot_all(self, data, key):
        for name in self.plotters().keys():
            self.plot_one(data, name, key)

    def plot_one(self, data, name, key):
        self.plotters()[name].build(data, key)


class LevenstheinDistancePlotter(Plotter):

    """creates a plot for every model, showing how correctness and edit-distance correlate
    """

    def build(self, data: dict, key: str) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between
            correctness and edit-distance for every model and configuration in the dictionary"""
        for model_class, models in data.items():
            for model, data in models.items():
                for config in ['original', 'paraphrased']:
                    results = data[config]['result']['data']
                    self._plot_single(results, model, config, key)


    def _plot_single(self, results: dict, model_name: str, config: str, key: str) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between"""
        plt.figure(figsize=(12,8))
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame({ 'distance': results['distance'], 'correct': results['correct']})
        # # Group the data by score and correctness to count the entries with the same score
        df_agg = df.groupby(['distance', 'correct']).size().reset_index(name='count')

        # Map the values of the 'correct' column to their respective terms
        df_agg['correctness'] = df_agg['correct'].replace({1: 'correct', 0: 'incorrect'})
        df_agg.drop('correct', axis=1, inplace=True)

        # Define the barplot using seaborn
        ax = sns.barplot(x='distance', y='count', hue='correctness', data=df_agg)

        # # Add labels and a title
        plt.xlabel('Score')
        plt.ylabel('Count')
        ax.legend(title="")
        plt.title('Results by Score and Correctness')
        plt.savefig(f"evaluation/plotting/plots/{key}_{model_name}_{config}_levensthein_distance.png")



class AccuracyOverviewPlotter(Plotter):

    """creates a plot including every models accuracies for all configs"""

    def build(self, data):
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
        plt.savefig(f"evaluation/plotting/plots/plot_{self.key}.png")


def main():
    # key for result dataset from command line arguments
    name, key, model = Plotter.parse_options()
    print("creating loader")
    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    if model is not None:
        results = loader.load(key, model)
    else:
        results = loader.load(key)

    computed = SinglePredictionEvaluator.compute_metrics(gt, results)
    plotter = Plotter()
    plotter.plot(name=name, data=computed, key=key)


if __name__ == "__main__":
    main()