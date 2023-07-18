import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
matplotlib.use('agg')

print(matplotlib.get_backend())

from evaluation.loader import ResultLoader
from evaluation.single_prediction_eval import SinglePredictionEvaluator
from evaluation.top_k_prediction_eval import TopKPredictionEvaluator

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
        # parser.add_argument("-e", "--exclude", help="Exclude specific models from the run. Format: model_name1,model_name2", type=str)
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
            matplotlib.pyplot.close()

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
                configs = (data.keys() & ["original", "paraphrased"])
                for config in configs:
                    # key might be present but without data if metrics were not computed
                    # for that config. in that case, just skip it
                    if not "result" in data[config].keys():
                        continue
                    results = data[config]['result']['data']
                    self._plot_single(results, f"{model_class}-{model}", config, key)


    def _plot_single(self, results: dict, model_name: str, config: str, key: str) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between"""
        plt.figure(figsize=(12,8))
        font_size = 12
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame({ 'distance': results['distance'], 'correct': results['correct']})
        # round the distance to the closest .1 decimal
        df['distance'] = df['distance'].round(1)
        # # Group the data by score and correctness to count the entries with the same score
        df_agg = df.groupby(['distance', 'correct']).size().reset_index(name='count')

        # Map the values of the 'correct' column to their respective terms
        df_agg['correctness'] = df_agg['correct'].replace({1: 'correct', 0: 'incorrect'})
        df_agg.drop('correct', axis=1, inplace=True)

        # Define the barplot using seaborn
        ax = sns.barplot(x='distance', y='count', hue='correctness', data=df_agg)

        # # Add labels and a title
        plt.xlabel('Levenshtein edit distance', fontsize=font_size)
        plt.ylabel('number of pages', fontsize=font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # plt.legend(fon)
        ax.legend(title="", fontsize=font_size, loc="upper right")
        plt.title(f"{model_name}", fontsize=font_size + 8)
        path = f"evaluation/plotting/plots/{key}_{model_name}_{config}_levensthein_distance.png"
        logging.info(f"saving to {path}")
        plt.savefig(path)



class AccuracyOverviewPlotter(Plotter):

    """creates a plot including every models accuracies for all configs"""

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
        # ensure pyplot does not run out of memory when too many plots are created
        matplotlib.pyplot.close()


def main():
    # key for result dataset from command line arguments
    name, key, model = Plotter.parse_options()
    assert key is not None, "No key provided"
    print("creating loader")
    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    if model is not None:
        results = loader.load(key, model)
    else:
        results = loader.load(key)


    configs = ['paraphrased']
    logging.info(f"Using configs {configs}. If you want to change this, adapt variable configs in plotter.py.")

    computed = TopKPredictionEvaluator.compute_metrics(gt, results, configs)
    plotter = Plotter()
    plotter.plot(name=name, data=computed, key=key)


if __name__ == "__main__":
    main()