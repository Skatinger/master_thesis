# plotting class which uses precomputed results

import sys
from evaluation.loader import ResultLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from adjustText import adjust_text

matplotlib.use('agg')

class PrecomputedPlotting():

    def __init__(self, key, model_name = None):
        self.size_label = "Size in Billion Parameters"
        self.accuracy_label = "Accuracy"
        print("loading results...")
        self.results = ResultLoader().load_computed(key, model_name)

    def plot(self):
        # convert results to dataframe format for easier plotting
        prepared_df = self.convert_to_df(self.results)
        self.plot_normal_to_instructional(self.results, prepared_df)
        self.plot_accuracy_progression(self.results, prepared_df)
        self.plot_best_performers(self.results, prepared_df)
        self.plot_with_huge(self.results, prepared_df)
        self.plot_accuracy_overview(self.results)
        self.plot_accuracy_overview_with_legend(self.results, prepared_df)
        self.plot_accuracy_input_size_comparison(self.results, prepared_df)
        self.plot_accuracy_overview_with_legend_and_size(self.results, prepared_df)
        self.tabulate_results_to_latex(self.results)

    @staticmethod
    def plot_with_huge(results, df2):
        # Extract the class of the model from the 'model' column
        my_font_size = 24

        # Find the row index with the highest accuracy for each model
        idx = df2.groupby('model_class')['accuracy'].idxmax()

        # Select the rows corresponding to the highest accuracy for each model
        best_rows = df2.loc[idx]
        df2 = best_rows

        plt.figure(figsize=(20, 14))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Add labels to each point with adjusted positions
        for i, row in df2.iterrows():
            label_length = len(row['model'])
            mv_left = label_length * 2 + -5
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                         fontsize=my_font_size)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Annotate the baselines
        plt.annotate("random names", (df2['size'].min(), 0.06), xytext=(-10, 5), textcoords='offset points', color='blue', fontsize=my_font_size)
        plt.annotate("majority names", (df2['size'].min(), 0.13), xytext=(-10, 5), textcoords='offset points', color='orange', fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0., fontsize='xx-large')

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_with_huge_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    @staticmethod
    def plot_accuracy_progression(results, df2):
        my_font_size = 24
        plt.figure(figsize=(20, 14))
        # only keep the interesting models
        interesting_models = ["bloomz", "flan_t5", "roberta", "roberta_squad", "t5"]

        # remove all models that are not in the interesting models list
        df2 = df2[df2['model_class'].isin(interesting_models)]

        # group it
        grouped_data = df2.groupby('model_class').apply(lambda x: x.sort_values('size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_class'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        # scatter points
        sns.scatterplot(data=grouped_data, x='size', y='accuracy', hue='model_class', s=250, markers=True, legend=False)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_class'] == group]
            plt.plot(group_data['size'], group_data['accuracy'], color=color, label=group, linewidth=3)

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Add legend
        legend = plt.legend(fontsize=my_font_size, framealpha=1)
        # increase linewidth of legend lines
        for line in legend.get_lines():
            line.set_linewidth(6)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_accuracy_progression_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def plot_best_performers(results, df2):
        # Extract the class of the model from the 'model' column
        my_font_size = 24

        # Find the row index with the highest accuracy for each model
        idx = df2.groupby('model_class')['accuracy'].idxmax()

        # Select the rows corresponding to the highest accuracy for each model
        best_rows = df2.loc[idx]
        df2 = best_rows

        plt.figure(figsize=(20, 14))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Add labels to each point with adjusted positions
        for i, row in df2.iterrows():
            label_length = len(row['model'])
            # if the point is to the right of the plot, move the label to the left
            if row['size'] == df2['size'].max():
                mv_left = - (label_length * 10)
            # small hack for gptj, to make it fit as well
            elif row['model_class'] == "gptj":
                mv_left = - (label_length * 8)
            # otherwise, move it to the right
            else:
                mv_left = 10
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(mv_left, -7), textcoords='offset points',
                         fontsize=my_font_size)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Annotate the baselines
        # plt.annotate("random names", (df2['size'].min(), 0.06), xytext=(-10, 5), textcoords='offset points', color='blue', fontsize=my_font_size)
        # plt.annotate("majority names", (df2['size'].min(), 0.13), xytext=(-10, 5), textcoords='offset points', color='orange', fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # with zero opacity
        # plt.legend(fontsize=my_font_size)
        plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)


        # plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def convert_to_df(results):
        # Create empty lists to store the data
        models = []
        sizes = []
        configs = []
        accuracies = []
        model_classes = []

        # Iterate over the dictionary and extract the data
        for model, model_data in results.items():
            if model == "key":
                continue

            # don't show baseline models
            if model.split("-")[0] in ["majority_full_name", "random_full_name"]:
                continue

            accuracies.append(model_data['paraphrased']["accuracy"])
            sizes.append(model_data['size'])
            models.append(model)
            configs.append('paraphrased')
            model_classes.append(model.split("-")[0])

        # # Create a DataFrame from the extracted data
        return pd.DataFrame({"model": models, "model_class": model_classes, "size": sizes, "accuracy": accuracies})

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

        plt.legend(fontsize="xx-large")


        # Set labels and title
        plt.xlabel("Size", fontsize="xx-large")
        plt.ylabel("Accuracy", fontsize="xx-large")

        # Increase font size of tick labels
        plt.xticks(fontsize='xx-large')
        plt.yticks(fontsize='xx-large')

        plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_{results['key']}.png", bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def plot_accuracy_overview_with_legend_and_size(results, df2):
        plt.figure(figsize=(20, 14))

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model_class", s=350, markers=True)

        # baselines
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize=my_font_size + 10)

        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize=my_font_size,
                #    markerscale=3.5)
        plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_and_size_{results['key']}.png") #, bbox_inches='tight')


    @staticmethod
    def plot_normal_to_instructional(results, df2):
        """expect df2 to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))

        # expect the following models
        models = ["falcon", "roberta", "t5", "distilbert"]

        df2["is_instructional"] = df2["model_class"].apply(
            lambda x: "instruction tuned" if ("squad" in x or "flan" in x or "instruct" in x) else "normal")

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="is_instructional", s=350, markers=True)

        for i, row in df2.iterrows():
            label_length = len(row['model'])
            # if the point is to the right of the plot, move the label to the left
            if row['size'] == df2['size'].max():
                mv_left = - (label_length * 10.5)
            # small hack for gptj, to make it fit as well
            else:
                mv_left = 12
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(mv_left, -7), textcoords='offset points',
                         fontsize=my_font_size)
    
        grouped_data = df2.groupby(df2["model"].str.extract(f"({'|'.join(models)})")[0])

        for group, group_data in grouped_data:

            # Sort the group data by 'size' in ascending order
            sorted_data = group_data.sort_values('is_instructional')
            
            # Extract the x and y values for the two points
            x1, y1 = sorted_data.iloc[0]['size'], sorted_data.iloc[0]['accuracy']
            x2, y2 = sorted_data.iloc[1]['size'], sorted_data.iloc[1]['accuracy']
            
            # padding to not overlap labels
            padding = 0.004
            # Calculate the difference in y values
            y_diff = abs(y2 - y1) - padding - 0.004 # remove dot radius

            start = min(y1, y2) + padding
            
            # Draw the arrow from normal to instructional
            plt.arrow(x1, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)
            
            # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
            percentage_length = 0.75
            plt.text(x1 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

        
        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        max_x = df2['size'].max()
        plt.xlim(None, max_x + percentage_length)


        legend = plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)
        # plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_normal_to_instructional_{results['key']}.png")


    @staticmethod
    def plot_accuracy_input_size_comparison(results, df2):
        """expect df2 to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))

        # use input size as hue
        df2['input_size'] = df2['model_class'].apply(lambda x: x.split("_")[-1])
        # remove input sizes from model class names
        df2['model_class'] = df2['model_class'].apply(lambda x: "_".join(x.split("_")[:-1]))

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="input_size", s=350, markers=True)

        for i, row in df2.iterrows():
            label_length = len(row['model'])
            mv_left = label_length * 2 + -5
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                         fontsize=my_font_size)
        

        grouped_data = df2.groupby('model_class')

        for group, group_data in grouped_data:

            # Sort the group data by 'size' in ascending order
            sorted_data = group_data.sort_values('size')
            
            # Extract the x and y values for the two points
            x1, y1 = sorted_data.iloc[0]['size'], sorted_data.iloc[0]['accuracy']
            x2, y2 = sorted_data.iloc[1]['size'], sorted_data.iloc[1]['accuracy']
            
            # padding to not overlap labels
            padding = 0.008
            # Calculate the difference in y values
            y_diff = abs(y2 - y1) - padding - 0.002 # remove dot radius

            start = min(y1, y2) + padding
            
            # Draw the arrow
            plt.arrow(x1, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)
            
            # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
            percentage_length = 0.8
            plt.text(x1 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        max_x = df2['size'].max()
        plt.xlim(None, max_x + 1)


        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize=my_font_size,
                #    markerscale=3.5)
        legend = plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)
        legend.set_title("Input Size [Chars]", prop={'size': my_font_size})
        # plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_input_size_comparison_{results['key']}.png")
    
    @staticmethod
    def plot_accuracy_overview_with_legend(results, df2):
        df2 = PrecomputedPlotting.convert_to_df(results)
        plt.figure(figsize=(20, 14))

        my_font_size = 24

        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=350, markers=True)

        # baselines
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize=my_font_size + 10)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2,fontsize=my_font_size,
                   markerscale=3.5)
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def tabulate_results_to_latex(results):
        df2 = PrecomputedPlotting.convert_to_df(results)
        df2 = df2.sort_values(by=['accuracy'], ascending=False)
        latext_text = df2.to_latex(index=False)
        with open(f"evaluation/plotting/plots/latex_table_{results['key']}.txt", "w") as f:
            f.write(latext_text)

def main():
    if len(sys.argv[1]) > 0:
        print(sys.argv[1])
        PrecomputedPlotting(sys.argv[1]).plot()
    else:
        PrecomputedPlotting("run-all-top-5").plot()


if __name__ == '__main__':
    main()
