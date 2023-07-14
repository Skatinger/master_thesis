import pandas as pd
import re


def compute_correctness(row, column):
    regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in str(row['urteil_angeklagter']).split()])
    return bool(re.match(regex, str(row[column]), re.IGNORECASE))

def main():
    # predictions laden
    predictions_data = pd.read_csv("legal-top-5-for-BG.csv")
    # Urteile laden
    actual_data = pd.read_csv("<pfad-zu-den-urteilen")
    # Tabellen verbinden
    merged_data = predictions_data.merge(actual_data, on='file_name', how='left')

    # für jedes Model die Korrektheit berechnen
    model_columns = ["legal_xlm_roberta-0b561", "legal_swiss_roberta-0b561", "bloomz-7b1", "legal_xlm_longformer-0b279",
                 "swiss_bert-0b110", "mt0-13b", "xlm_swiss_bert-0b110"]

    for column in model_columns:
        merged_data[f"{column}_correct"] = merged_data.apply(lambda row: compute_correctness(row, column), axis=1)

    # Score berechnen
    accuracy = {}
    for column in model_columns:
        accuracy[column] = merged_data[f"{column}_correct"].mean() * 100  # convert to percentage

    # Spalten mit Korrektheit zur Anonymisierung der Auswertung wieder entfernen
    merged_data = merged_data.drop(columns=[f"{column}_correct" for column in model_columns])

    # Score für jedes Model ausgeben
    for model, acc in accuracy.items():
        print(f"Accuracy for {model}: {acc:.2f}%")
