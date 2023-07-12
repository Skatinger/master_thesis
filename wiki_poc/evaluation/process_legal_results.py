import sys
import pandas as pd
from evaluation.loader import ResultLoader
from datasets import load_dataset
from names_dataset import NameDataset

name_dataset = NameDataset()

def is_name(word: str) -> bool:
    """checks if a word is a name by checking if it is in the names dataset"""
    return name_dataset.search(word)["first_name"] is not None or name_dataset.search(word)["last_name"] is not None

def join_predictions(row):
    row = row.to_dict()
    string = ""
    for key, value in row.items():
        if key.startswith("prediction") and value not in row["masked_text_original"]:
            if is_name(value):
                string += f" {value}"
    return string

def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    key = sys.argv[1]

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
    predictions = loader.load(key)
    
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

            # add a column with the model name and size and fill it with the joined predictions
            predictions_df[f"{model_class_name}-{model_size}"] = predictions_df.apply(join_predictions, axis=1)

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


if __name__ == "__main__":
    main()