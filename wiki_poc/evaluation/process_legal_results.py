import sys
import pandas as pd
import numpy as np
from evaluation.loader import ResultLoader
from datasets import load_dataset
from names_dataset import NameDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

name_dataset = NameDataset()

global classification
classification = {
        "exist_in_text": {},
        "not_a_name": {},
        "just_a_letter": {},
        "good": {},
    }

def is_name(word: str) -> bool:
    """checks if a word is a name by checking if it is in the names dataset"""
    return name_dataset.search(word)["first_name"] is not None or name_dataset.search(word)["last_name"] is not None

def join_predictions(row, name):

    # dict to track which values are what category (present in text, not a name, good)
    row = row.to_dict()
    string = ""
    for key, value in row.items():
        if key.startswith("prediction") and value not in row["masked_text_original"]:
            if is_name(value):
                string += f" {value}"
                classification["good"][name] += 1
            else:
                # 2 letters or less without whitespace are not names
                if len(value.replace(" ", "")) < 3:
                    classification["just_a_letter"][name] += 1
                else:
                    classification["not_a_name"][name] += 1
        else:
            # value is already in masked_text_original
            classification["exist_in_text"][name] += 1
    return string

def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    dataset_key = sys.argv[1]

    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_rulings_gt()
    gt_df = gt.to_pandas()

    print("loading rulings dataset")
    rulings = load_dataset("rcds/swiss_rulings", split="train")
    # only keep rulings with decision_id in gt ids
    gt_ids = gt["id"]
    rulings = rulings.filter(lambda x: x['decision_id'] in gt_ids)

    print("loading predictions")
    predictions = loader.load(dataset_key)
    
    full_rulings_df = rulings.to_pandas()
    # Rename the column "decision_id" to "page_id" in full_rulings_df to match with other dataframes
    full_rulings_df = full_rulings_df.rename(columns={"decision_id": "page_id"})

    model_columns = []
    for model_class_name, model_class in predictions.items():
        for model_size, model in model_class.items():
            # Convert the predictions dataset to a pandas dataframe
            predictions_df = model['original']['train'].to_pandas()

            # join the ground truth onto the predictions
            predictions_df = pd.merge(predictions_df, gt_df, left_on="page_id", right_on="id")

            # prepare classification dict
            for key in classification.keys():
                classification[key][f"{model_class_name}-{model_size}"] = 0

            # add a column with the model name and size and fill it with the joined predictions
            model_name = f"{model_class_name}-{model_size}"
            predictions_df[f"{model_class_name}-{model_size}"] = predictions_df.apply(join_predictions, name=model_name, axis=1)

            # join the predictions dataframe to the rulings dataframe on "page_id"
            full_rulings_df = pd.merge(full_rulings_df, predictions_df[["page_id", f"{model_class_name}-{model_size}"]], on="page_id")

            # add the model name and size to the list of columns
            model_columns.append(f"{model_class_name}-{model_size}")

    
    # only keep the columns we need for the final dataframe
    final_df = full_rulings_df[["page_id", "file_id", "file_name", "file_number"] + model_columns]

    # Create a new column 'total_length' that contains the sum of the character lengths of the last columns
    final_df['total_length'] = final_df[model_columns].apply(lambda row: sum(row.fillna('').apply(len)), axis=1)

    # Sort the DataFrame based on the 'total_length' column
    final_df = final_df.sort_values(by='total_length', ascending=False)

    # Delete the 'total_length' column as it's no longer needed
    del final_df['total_length']

    # Save the final dataframe to a CSV file
    final_df.to_csv(f"{key}-for-BG.csv", index=False)

    # plot a histogram of the lengths of classification categories
    labels = list(classification.keys())
    models = list(classification[labels[0]].keys())

    # Calculate total scores for each model and sort models accordingly
    total_scores = {model: sum(classification[label][model] for label in labels) for model in models}
    models.sort(key=lambda x: total_scores[x], reverse=True)

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects = []
    for i, model in enumerate(models):
        model_values = [classification[label][model] for label in labels]
        rects.append(ax.bar(x - width*len(models)/2 + i*width, model_values, width, label=model))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Type of prediction by group and model', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(fontsize=12)

    # Move x-tick labels to the center of the sections
    for tick in ax.get_xticklabels():
        tick.set_ha('center')

    plt.xlabel('Classified as', fontsize=18)
    plt.ylabel('number of predictions', fontsize=18)

    plt.savefig(f"{dataset_key}-histogram.png")

if __name__ == "__main__":
    main()