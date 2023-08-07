# plotting class which uses precomputed results

import sys
from evaluation.loader import ResultLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import matplotlib
from adjustText import adjust_text
from collections import defaultdict


matplotlib.use('agg')

class PrecomputedPlotting():

    def __init__(self, key, model_name = None):
        self.size_label = "Size in Billion Parameters"
        self.accuracy_label = "Accuracy"
        print("loading results...")
        self.results = ResultLoader().load_computed(key, model_name)
        self.set_baseline_values()

    
    def set_baseline_values(self):
        logging.info("setting baseline values, using: ")
        # hardcoded baselines for weighted scores
        self.baseline_random = 0.03
        self.baseline_majority = 0.11
        logging.info(f"random_full_name: {self.baseline_random}")
        logging.info(f"majority_full_name: {self.baseline_majority}")

    def plot(self):
        # convert results to dataframe format for easier plotting
        prepared_df = self.convert_to_df(self.results)
        # self.plot_normal_to_instructional(self.results, prepared_df)
        # self.plot_normal_to_instructional_barplot(self.results, prepared_df)
        self.plot_accuracy_progression(self.results, prepared_df)
        # self.plot_best_performers(self.results, prepared_df)
        # self.plot_with_huge(self.results, prepared_df)
        # self.plot_accuracy_overview(self.results)
        # self.plot_accuracy_overview_with_legend(self.results, prepared_df)
        # self.plot_accuracy_input_size_comparison(self.results, prepared_df)
        # self.input_length_progression(self.results, prepared_df)
        # self.sampling_method_comparison(self.results, prepared_df)
        # self.plot_accuracy_overview_with_legend_and_size(self.results, prepared_df)
        # self.plot_model_types_comparison(self.results, prepared_df)
        # self.plot_model_types_comparison_scatter(self.results, prepared_df)
        # self.compute_difference_paraphrased_to_original(self.results, prepared_df)

    def compute_difference_paraphrased_to_original(self, results, prepared_df):
        # Define metrics to be calculated
        metrics = ["accuracy", "precision", "last_name_accuracy", "last_name_precision", "weighted_score"]

        # Initialize counters and sums for original and paraphrased configurations
        counters = defaultdict(int)
        sums = defaultdict(lambda: defaultdict(float))

        # Iterate through the data and calculate the sums and counts
        for model, model_data in results.items():
            if model == "key":
                continue
            for config in ["original", "paraphrased"]:
                if config not in model_data.keys():
                    continue
                # if model_data[config]["weighted_score"] < self.baseline_majority:
                #     continue
                counters[config] += 1
                for metric in metrics:
                    sums[config][metric] += model_data[config][metric]

        # Calculate averages
        averages = defaultdict(lambda: defaultdict(float))
        for config in ["original", "paraphrased"]:
            for metric in metrics:
                averages[config][metric] = sums[config][metric] / counters[config] if counters[config] else 0

        # Create a table with the results
        table_data = [["Configuration"] + metrics]
        for config in ["original", "paraphrased"]:
            table_data.append([config] + [averages[config][metric] for metric in metrics])
        
        # convert table_data to dataframe
        table_data = pd.DataFrame(table_data[1:], columns=table_data[0])

        # save result to file
        latext_text = table_data.to_latex(index=False, float_format="{:0.2f}".format)
        with open(f"evaluation/plotting/plots/text_results/paraphrased_to_original_{results['key']}.tex", "w") as f:
            f.write(latext_text)

    @staticmethod
    def sampling_method_comparison(results, _df):
        data_for_df = []

        for key, value in results.items():
            if key != 'key':
                method = key.split('--')[-1]
                if method == "beam_search_sampling":
                    method = "beam-search"
                elif method == "sampling":
                    method = "top-k\nsampling"
                elif method == "beam_search":
                    method = "beam search\n(n_beams=5)"
                elif method == "top_p_sampling":
                    method = "top-p\n(p=0.92)"
                elif method == "top_k_sampling":
                    method = "top-k\n(k=50)"
                elif method == "nucleus_sampling":
                    method = "nucleus\n(k=50, p=0.92)"
                elif method == "random_sampling":
                    method = "random"
                elif method == "greedy":
                    method = "greedy\n(top-1)"
                elif method == "top_k_sampling_kruns":
                    method = "top-k\n(k=5)"
                # accuracy = value['paraphrased']['accuracy']
                weighted_score = value['paraphrased']['weighted_score']
                # data_for_df.append([method, accuracy])
                data_for_df.append([method, weighted_score])

        df = pd.DataFrame(data_for_df, columns=['method', 'weighted_score'])
        df = df.sort_values(by=['weighted_score'], ascending=False)

        ax = df.plot(kind='bar', x='method', y='weighted_score', legend=False)
        plt.ylabel('W-PNMS', fontsize=16)
        # plt.title('Comparison of Different Generation Methods (top 5)\nincite_instruct-3b')
        plt.xticks(rotation=45, ha='right')
        # remove x axis label
        # increase font size of labels
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('')
        plt.tight_layout()  # adjusts subplot params so that the subplot fits into the figure area

        plt.savefig('evaluation/plotting/plots/ablations/sampling_method_comparison.png', dpi=400, bbox_inches='tight')

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
        plt.axhline(y=self.random_baseline, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=self.majority_baseline, color='orange', linewidth=2.5, label="majority names")

        # Annotate the baselines
        plt.annotate("random names", (df2['size'].min(), 0.06), xytext=(-10, 5), textcoords='offset points', color='blue', fontsize=my_font_size)
        plt.annotate("majority names", (df2['size'].min(), 0.13), xytext=(-10, 5), textcoords='offset points', color='orange', fontsize=my_font_size)

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Set x-axis to logarithmic scale
        plt.xscale('log')
        # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0., fontsize='xx-large')

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_with_huge_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    @staticmethod
    def plot_accuracy_progression(results, df2):
        my_font_size = 42
        plt.figure(figsize=(20, 14))
        # only keep the interesting models, e.g. models which were evaluated in different sizes
        interesting_models = ["bloomz", "flan_t5", "roberta", "t5", "mt0", "bloom", "cerebras", "pythia", "t0"]
                              # , "llama2"] # ditched roberta_squad and llama2, not interesting

        # remove all models that are not in the interesting models list
        df2 = df2[df2['model_class'].isin(interesting_models)]

        # remove any models above 20 billion parameters, as they are not interesting because
        # they pull apart the plot too much
        df2 = df2[df2['size'] < 20]

        # group it
        grouped_data = df2.groupby('model_class').apply(lambda x: x.sort_values('size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_class'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        # scatter points
        sns.scatterplot(data=grouped_data, x='size', y='weighted_score', hue='model_class', s=500, markers=True, legend=False)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_class'] == group]
            plt.plot(group_data['size'], group_data['weighted_score'], color=color, label=group, linewidth=7)

        # Set labels and title
        plt.xlabel("model size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("W-PNMS", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Add legend, place it outside of plot
        legend = plt.legend(fontsize=my_font_size - 3, framealpha=1) #, loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0.)
        # increase linewidth of legend lines
        for line in legend.get_lines():
            line.set_linewidth(8)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_accuracy_progression_{results['key']}.png", bbox_inches='tight', dpi=300)
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def input_length_progression(results, df):
        my_font_size = 42
        plt.figure(figsize=(16, 10))

        # use input size as hue
        df['input_size'] = df['model_class'].apply(lambda x: x.split("_")[-1])
        # remove input sizes from model class names
        df['model_class'] = df['model_class'].apply(lambda x: "_".join(x.split("_")[:-1]))

        # group it
        df = df.sort_values('input_size', ascending=False)
        grouped_data = df.groupby('model_class').apply(lambda x: x.sort_values('input_size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_class'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        # scatter points
        sns.scatterplot(data=grouped_data, x='input_size', y='accuracy', hue='model_class', s=450, markers=True, legend=False)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_class'] == group]
            plt.plot(group_data['input_size'], group_data['accuracy'], color=color, label=group, linewidth=9)

        # Set labels and title
        plt.xlabel("input size [characters]", fontsize=my_font_size)
        plt.ylabel("PNMS", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Add legend
        legend = plt.legend(fontsize=my_font_size, framealpha=1)
        # increase linewidth of legend lines
        for line in legend.get_lines():
            line.set_linewidth(9)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_input_length_progression_{results['key']}.png", bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    def plot_wiki_edits_proxy(results, df2):
        """Plot the wiki edits proxy metric for each model
           this shows the number of edits to a page compared to the length of the page"""



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
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # scale x as log
        plt.xscale('log')

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
        precisions = []
        weighted_scores = []
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
            precisions.append(model_data['paraphrased']["precision"])
            model_classes.append(model.split("-")[0])
            if "weighted_score" in model_data['paraphrased']:
                weighted_scores.append(model_data['paraphrased']["weighted_score"])
            else:
                logging.warning(f"no weighted score for {model}")
                weighted_scores.append(0)
                

        # # Create a DataFrame from the extracted data
        return pd.DataFrame({"model": models, "model_class": model_classes, "size": sizes, "precision": precisions,
                             "accuracy": accuracies, "weighted_score": weighted_scores, "config": configs})

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

    def plot_model_types_comparison_scatter(self, results, df):
        # add slighly larger fontsize
        my_font_size = 12
        # Dict
        model_types = {
            "bert": "fill_mask",
            "bloom": "text_generation",
            "bloomz": "text_generation",
            "cerebras": "text_generation",
            "deberta": "fill_mask",
            "deberta_squad": "question_answering",
            "distilbert_squad": "question_answering",
            "distilbert": "fill_mask",
            "falcon": "text_generation",
            "falcon_instruct": "text_generation",
            "flan_t5": "text_generation",
            "gptj": "text_generation",
            "incite_instruct": "text_generation",
            "llama": "text_generation",
            "mpt": "text_generation",
            "mt0": "text_generation",
            "mt5": "text_generation",
            "pythia": "text_generation",
            "roberta_squad": "question_answering",
            "t5": "text_generation",
            "roberta": "fill_mask",
            "gpt3.5turbo": "text_generation",
            "gpt_4": "text_generation",
        }

        # Convert dict to DataFrame
        model_types_df = pd.DataFrame(list(model_types.items()), columns=['model_class', 'model_type'])

        # Merge dataframes on model_class
        df = pd.merge(df, model_types_df, on='model_class')

        # Plot
        plt.figure(figsize=(10, 8))
        print(df)

        # group it
        df = df.sort_values('size', ascending=False)
        grouped_data = df.groupby('model_type').apply(lambda x: x.sort_values('size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_type'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        my_font_size = 24

        # scatter points
        scatter = sns.scatterplot(data=grouped_data, x='size', y='weighted_score', hue='model_type', s=250, markers=True, legend=True)
        handles, labels = scatter.get_legend_handles_labels()

        # baselines
        plt.axhline(y=self.baseline_random, color='blue', linewidth=2.5, label="random names", zorder=0)
        plt.axhline(y=self.baseline_majority, color='orange', linewidth=2.5, label="majority names", zorder=0)

        # Create custom legend entries
        import matplotlib.lines as mlines
        # random_patch = mpatches.Patch(color='blue', label='random names')
        # majority_patch = mpatches.Patch(color='orange', label='majority names')
        random_line = mlines.Line2D([], [], color='blue', marker='_', markersize=15, label='random names baseline', linewidth=1.5)
        majority_line = mlines.Line2D([], [], color='orange', marker='_', markersize=15, label='majority names baseline', linewidth=1.5)


        # handles.extend([random_patch, majority_patch])
        handles.extend([random_line, majority_line])
        labels.extend(['random names', 'majority names'])

        plt.legend(handles=handles, labels=labels)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_type'] == group]
            plt.plot(group_data['size'], group_data['weighted_score'], color=color, label="_nolegend_", linewidth=3)



        # sns.scatterplot(data=df, x="size", y="weighted_score", hue="model_type", s=250, markers=True, legend=True)

        plt.legend(fontsize="x-large")
        plt.xlabel('model size [million parameters]', fontsize="x-large")
        plt.ylabel('weighted partial name match score', fontsize="x-large")
        plt.title('Model Performance by Type', fontsize="x-large")
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_model_types_comparison_scatter_{results['key']}.png")


    
    def plot_model_types_comparison(self, results, df):

        # Dict
        model_types = {
            "bert": "fill_mask",
            "bloom": "text_generation",
            "bloomz": "text_generation",
            "cerebras": "text_generation",
            "deberta": "fill_mask",
            "deberta_squad": "question_answering",
            "distilbert_squad": "question_answering",
            "distilbert": "fill_mask",
            "falcon": "text_generation",
            "falcon_instruct": "text_generation",
            "flan_t5": "text_generation",
            "gptj": "text_generation",
            "incite_instruct": "text_generation",
            "llama": "text_generation",
            "mpt": "text_generation",
            "mt0": "text_generation",
            "mt5": "text_generation",
            "pythia": "text_generation",
            "roberta_squad": "question_answering",
            "t5": "text_generation",
            "roberta": "fill_mask",
            "gpt3.5turbo": "text_generation",
            "gpt_4": "text_generation",
        }

        # Convert dict to DataFrame
        model_types_df = pd.DataFrame(list(model_types.items()), columns=['model_class', 'model_type'])

        # Merge dataframes on model_class
        df = pd.merge(df, model_types_df, on='model_class')

        # Apply log transformation to 'size' column
        df['log_size'] = np.log1p(df['size'])

        # Divide 'log_size' into quantiles
        bins = [0.1,0.5,1]
        df['size_group'] = pd.cut(df['size'], bins=bins)
        print(df)

        # Sorting by size_group and weighted_score
        df.sort_values(by=['weighted_score', 'size'], inplace=True)

        # Convert size_group to string for plotting
        df['size_group'] = 'Group ' + pd.cut(df['size'], bins=bins).cat.codes.astype(str)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='accuracy', y='model', hue='model_type', data=df, dodge=False)

        # Adding a vertical line to separate the groups
        unique_groups = df['size_group'].unique()
        for i in range(1, len(unique_groups)):
            plt.axhline(y=df[df['size_group'] == unique_groups[i]].index.min()-0.5, color='black', linestyle='--')

        plt.xlabel('Accuracy')
        plt.ylabel('Model')
        plt.title('Model Performance by Type')
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_model_types_comparison_{results['key']}_2.png")


    def plot_accuracy_overview_with_legend_and_size(self, results, df2):
        print(df2.head())
        plt.figure(figsize=(20, 14))

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="weighted_score", hue="model_class", s=350, markers=True)

        # baselines
        plt.axhline(y=self.baseline_random, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=self.baseline_majority, color='orange', linewidth=2.5, label="majority names")

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
        models = ["falcon", "roberta", "distilbert", "mpt", "incite", "t5"]

        print("at least here")

        df2["is_instructional"] = df2["model_class"].apply(
            lambda x: "instruction tuned" if ("squad" in x or "flan" in x or "instruct" in x) else "base")

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="weighted_score", hue="is_instructional", s=350, markers=True)

        for i, row in df2.iterrows():
            label_length = len(row['model'])
            # if the point is to the right of the plot, move the label to the left
            if row['size'] == df2['size'].max():
                mv_left = - (label_length * 10.5)
            # small hack for gptj, to make it fit as well
            else:
                mv_left = 12
            plt.annotate(row['model_class'], (row['size'], row['weighted_score']), xytext=(mv_left, -7), textcoords='offset points',
                         fontsize=my_font_size)

        grouped_data = df2.groupby(df2["model"].str.extract(f"({'|'.join(models)})")[0])

        for group, group_data in grouped_data:

            # Sort the group data by 'size' in ascending order
            sorted_data = group_data.sort_values('is_instructional')
            
            # Extract the x and y values for the two points
            x1, y1 = sorted_data.iloc[0]['size'], sorted_data.iloc[0]['weighted_score']
            x2, y2 = sorted_data.iloc[1]['size'], sorted_data.iloc[1]['weighted_score']
            
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

        plt.savefig(f"evaluation/plotting/plots/ablations/plot_normal_to_instructional_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def plot_normal_to_instructional_barplot(results, df):
        """expect df to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))
        my_font_size = 42

        # expect the following models
        models = ["falcon", "roberta", "distilbert", "mpt", "incite", "t5", "bloom"]

        # Group by model names
        df['model_name'] = df["model"].str.extract(f"({'|'.join(models)})")[0]

        df["is_instructional"] = df["model_class"].apply(
            lambda x: "instruction tuned" if ("squad" in x or "flan" in x or "instruct" in x) else "base")

        grouped = df.groupby(['model_name', 'is_instructional'])
        size_grouped = df.groupby('model_name')['size'].mean()

        # Prepare data for plot
        plot_data = []
        for name, group in grouped:
            plot_data.append((name[0], name[1], group['weighted_score'].mean()))

        # Sort data by score
        plot_data.sort(key=lambda x: x[2])

        # Split data into separate lists
        names, types, accuracies = zip(*plot_data)
        unique_names = sorted(set(names))
        model_sizes = df.groupby('model_name')['size'].unique()

        # Create x-axis tick labels with size information
        unique_names_with_size = []
        sizes_in_billion = []
        for name in unique_names:
            sizes = []
            for size in model_sizes[name]:
                if size < 1:
                    # Format as millions with non-zero decimals
                    sizes.append(f'{size*1000:.0f}M' if size*1000 % 1 == 0 else f'{size*1000:.1f}M')
                else:
                    # Format as billions with non-zero decimals
                    sizes.append(f'{size:.0f}B' if size % 1 == 0 else f'{size:.1f}B')
            unique_names_with_size.append(f'{name}\n{"/".join(sizes)}')
            sizes_in_billion.append(float(sizes[0][:-1]) if sizes[0][-1] == 'B' else float(sizes[0][:-1]) / 1000)

        # Combine the names and sizes, and sort them by size
        names_and_sizes = sorted(zip(unique_names_with_size, sizes_in_billion), key=lambda x: x[1])

        # Get the sorted names
        unique_names_with_size = [name for name, size in names_and_sizes]

        normal_accuracies = [accuracy for name, type_, accuracy in plot_data if type_ == 'base']
        instruction_accuracies = [accuracy for name, type_, accuracy in plot_data if type_ == 'instruction tuned']

        # Calculate the width of a bar
        bar_width = 0.3

        # Positions of the left bar boundaries
        bar_l = np.arange(len(unique_names))

        # Positions of the x-axis ticks (center of the bars as bar labels)
        tick_pos = [i + bar_width / 2 for i in bar_l]

        # Create the bar plot
        plt.bar(bar_l, normal_accuracies, width=bar_width, label='base', color='orange')
        plt.bar(bar_l + bar_width, instruction_accuracies, width=bar_width, label='instructional', color='blue')

        # Set the labels and title
        plt.ylabel('W-PNMS', fontsize=my_font_size)

        # Set the positions and labels of the x-axis ticks
        plt.xticks(tick_pos, unique_names_with_size)
        
        # increase height of plot so legend fits
        y_max = max(max(normal_accuracies), max(instruction_accuracies))
        plt.ylim(0, y_max + 0.07)

        # Adding the legend and showing the plot
        plt.legend(fontsize=my_font_size)
        
        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        plt.savefig(f"evaluation/plotting/plots/ablations/plot_normal_to_instructional_barplot_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def plot_accuracy_input_size_comparison(results, df2):
        """expect df2 to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))

        # use input size as hue
        df2['input_size'] = df2['model_class'].apply(lambda x: x.split("_")[-1])
        # remove input sizes from model class names
        df2['model_class'] = df2['model_class'].apply(lambda x: "_".join(x.split("_")[:-1]))

        my_font_size = 24
        df3 = df2.sort_values('input_size', ascending=False)
        sns.scatterplot(data=df3, x="size", y="accuracy", hue="input_size", s=350, markers=True)

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
            y_diff = abs(y2 - y1) - 0.006 # remove dot radius

            start = min(y1, y2) + 0.003
            if len(sorted_data) == 2:
                start += padding
                y_diff -= padding
            
            # Draw the arrow
            plt.arrow(x1, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)
            
            # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
            percentage_length = 0.86
            plt.text(x1 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

            # got a third point, compare it as well
            if len(sorted_data) == 3:
                x3, y3 = sorted_data.iloc[2]['size'], sorted_data.iloc[2]['accuracy']

                # Calculate the difference in y values
                y_diff = abs(y3 - y1) - 0.006 # remove dot radius

                start = min(y1, y3) + 0.003

                # Draw the arrow
                plt.arrow(x2, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)

                # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
                percentage_length = 0.8
                plt.text(x2 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

                # add the model label
                entry = sorted_data.iloc[2]
                label_length = len(entry['model'])
                # only label the top-most
                mv_left = label_length * 2 + -5
                plt.annotate(entry['model_class'], (entry['size'], entry['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                            fontsize=my_font_size)
            else:
                # add the model label
                entry = sorted_data.iloc[1]
                label_length = len(entry['model'])
                # only label the top-most
                mv_left = label_length * 2 + -5
                plt.annotate(entry['model_class'], (entry['size'], entry['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                            fontsize=my_font_size)

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        max_x = df2['size'].max()
        plt.xlim(None, max_x + 1)

        legend = plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)
        legend.set_title("input size [chars]", prop={'size': my_font_size})

        plt.savefig(f"evaluation/plotting/plots/plot_input_length_comparison_{results['key']}.png")
    
    @staticmethod
    def plot_accuracy_overview_with_legend(results, df2):
        plt.figure(figsize=(20, 14))

        my_font_size = 24

        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=350, markers=True)

        # baselines
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Set x-axis to logarithmic scale
        plt.xscale('log')

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize=my_font_size + 10)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2,fontsize=my_font_size,
                   markerscale=3.5)
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def tabulate_results_to_latex(results):
        df2 = PrecomputedPlotting.convert_to_df(results)        
        # Keep only the specified columns
        df2 = df2[['model_class', 'size', 'accuracy', 'precision', 'weighted_score']]
        
        df2 = df2.sort_values(by=['weighted_score'], ascending=False)
        # round and reduce accuracy to 2 digits
        df2['accuracy'] = df2['accuracy'].apply(lambda x: round(x, 2))
        df2['size'] = df2['size'].apply(lambda x: round(x, 2))
        df2['precision'] = df2['precision'].apply(lambda x: round(x, 2))
        df2['weighted_score'] = df2['weighted_score'].apply(lambda x: round(x, 2))

        # Rename the columns
        df2 = df2.rename(columns={'size': 'size [billions]', 'accuracy': 'partial name matching score',
                                  'precision': 'levenshtein distance','weighted_score': 'weighted score'})
    
        # Ensure the index is of string type
        df2.index = df2.index.astype(str)
        
        # Now you can replace underscores
        df2.index = df2.index.str.replace('_', '\\_')
        
        latext_text = df2.to_latex(index=False, float_format="{:0.2f}".format)
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
