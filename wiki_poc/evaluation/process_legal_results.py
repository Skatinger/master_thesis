import sys
import pandas as pd
from evaluation.loader import ResultLoader
from datasets import load_dataset

def is_name(word: str) -> bool:
    return word.istitle()

def join_predictions(row):
    row = row.to_dict()
    string = ""
    for key, value in row.items():
        if key.startswith("prediction") and value not in row["masked_text_original"]:
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
    predictions = loader.load(key, model_name="bloomz-7b1")
    # and convert to pandas
    predictions_df = predictions['bloomz']['7b1']['original']['train'].to_pandas()

    full_rulings_df = rulings.to_pandas()
    # Rename the column "decision_id" to "page_id" in full_rulings_df to match with other dataframes
    full_rulings_df = full_rulings_df.rename(columns={"decision_id": "page_id"})

    # Merge the predictions dataframe with the rulings test dataframe on "page_id" and "id" respectively
    merged_df = pd.merge(predictions_df, gt_df, left_on="page_id", right_on="id")
    # Remove predictions from the "joined" column that exist in the "masked_text_original" column
    merged_df["joined"] = merged_df.apply(join_predictions, axis=1)

    # Merge the modified predictions dataframe with the full rulings dataframe on "page_id"
    final_df = pd.merge(merged_df[["page_id", "joined"]], full_rulings_df[["page_id", "file_id", "file_name", "file_number"]], on="page_id")

    # Save the final dataframe to a CSV file
    final_df.to_csv("final_output.csv", index=False)


if __name__ == "__main__":
    main()