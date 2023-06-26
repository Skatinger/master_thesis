# plotting class which uses precomputed results

from evaluation.loader import ResultLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from adjustText import adjust_text

matplotlib.use('agg')

class PrecomputedPlotting():

    def __init__(self, key, model_name = None):
        self.results = ResultLoader().load_computed(key, model_name)

    def plot(self):
        self.plot_best_performers(self.results)
        self.plot_accuracy_overview(self.results)
        self.plot_accuracy_overview_with_legend(self.results)
        self.tabulate_results_to_latex(self.results)

    @staticmethod
    def plot_best_performers(results):
        df2 = PrecomputedPlotting.convert_to_df(results)

        # Extract the class of the model from the 'model' column
        df2['model_class'] = df2['model'].str.split('-').str[0]

        # Find the row index with the highest accuracy for each model
        idx = df2.groupby('model_class')['accuracy'].idxmax()

        # Select the rows corresponding to the highest accuracy for each model
        best_rows = df2.loc[idx]
        df2 = best_rows

        plt.figure(figsize=(20, 16))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Add labels to each point with adjusted positions
        for i, row in df2.iterrows():
            label_length = len(row['model'])
            mv_left = label_length * 2
            plt.annotate(row['model'], (row['size'], row['accuracy']), xytext=(-mv_left, 15), textcoords='offset points', fontsize='large')

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size", fontsize="x-large")
        plt.ylabel("Accuracy", fontsize="x-large")

        # Increase font size of tick labels
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')

        plt.title("accuracy compared to model size\nfor paraphrased texts")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_{results['key']}.png")
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def convert_to_df(results):
        # Create empty lists to store the data
        models = []
        sizes = []
        configs = []
        accuracies = []

        # Iterate over the dictionary and extract the data
        for model, model_data in results.items():
            if model == "key":
                continue

            accuracies.append(model_data['paraphrased']["accuracy"])
            sizes.append(model_data['size'])
            models.append(model)
            configs.append('paraphrased')

        # # Create a DataFrame from the extracted data
        return pd.DataFrame({"model": models, "size": sizes, "accuracy": accuracies})

    @staticmethod
    def plot_accuracy_overview(results):
        df2 = PrecomputedPlotting.convert_to_df(results)

        plt.figure(figsize=(20, 16))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Prepare the labels and positions
        labels = df2['model'].tolist()
        positions = df2[['size', 'accuracy']].values.tolist()

        # Add labels with adjustment to avoid overlap
        texts = []
        for label, position in zip(labels, positions):
            label_length = len(label)
            mv_left = label_length * 2
            text = plt.annotate(label, position, xytext=(-mv_left, 15), textcoords='offset points', fontsize='large')
            texts.append(text)

        # Adjust the positions of labels to prevent overlap
        adjust_text(texts)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size", fontsize="x-large")
        plt.ylabel("Accuracy", fontsize="x-large")

        # Increase font size of tick labels
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')

        plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_{results['key']}.png")
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    @staticmethod
    def plot_accuracy_overview_with_legend(results):
        df2 = PrecomputedPlotting.convert_to_df(results)
        plt.figure(figsize=(20, 16))

        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True)

        # baselines
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size", fontsize="x-large")
        plt.ylabel("Accuracy", fontsize="x-large")

        # Increase font size of tick labels
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')

        plt.title("accuracy compared to model size\nfor paraphrased texts")

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=1,fontsize="large")
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_{results['key']}.png")

    @staticmethod
    def tabulate_results_to_latex(results):
        df2 = PrecomputedPlotting.convert_to_df(results)
        df2 = df2.sort_values(by=['accuracy'], ascending=False)
        latext_text = df2.to_latex(index=False)
        with open(f"evaluation/plotting/plots/latex_table_{results['key']}.txt", "w") as f:
            f.write(latext_text)

def main():
    PrecomputedPlotting("run-all-top-5").plot()


if __name__ == '__main__':
    main()
