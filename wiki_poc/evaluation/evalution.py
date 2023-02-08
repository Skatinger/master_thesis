# DOC
# takes the top 5 predictions for every mask,
# checks if any of the top 5 predictions matches part of the entity which was masked
# if so, it counts as correct match.
# recall is not an issue yet


# from unittest import result
import pandas as pd
import os
import logging
from tqdm import tqdm
from ast import literal_eval

resultsFile = 'wiki-dataset-results-1k.csv'


# results data is not always in same format, ensure it's always list of list
def reformat(data):
    results = []
    for prediction in data:
        if not any(isinstance(el, list) for el in prediction):
            results.append([prediction])  # transform it
        else:
            results.append(prediction)
    return results


# retrieves only the predicted tokens, without scoring and or other attributes
def extractPredictedTokens(data):
    predicted = []
    for predictions in data:
        for masked_word_predictions in predictions:
            tokens = [pred['token_str'] for pred in masked_word_predictions]
            predicted.append(tokens)
    return predicted


# returns the percentage of correct predictions
# correct means: an entity we want is within the top 5 predictions, see doc at top of file
def scoreMatches(predictions, entities):
    entitiesString = " ".join(entities)
    correctCount = 0
    for pred in predictions:
        if any([p in entitiesString for p in pred]):
            correctCount += 1
    if len(predictions) == 0:
        return 0
    else:
        return 100 / len(predictions) * correctCount


if __name__ == '__main__':

    # Import Data from CSV
    file = resultsFile
    assert len(file) > 0, "Please provide a file path to the dataset"

    if not os.path.exists(file):
        logging.error("Input file does not exist. Please run `demasking.py` first.")
        quit()

    print("loading dataset")
    dataset = pd.read_csv(file)

    # format predictions, so that all have the same format (sometimes its [[]], other times only [])
    tqdm.pandas()
    dataset['normal_predictions'] = dataset['normal_predictions'].apply(lambda data: reformat(literal_eval(data)))
    dataset['paraphrased_predictions'] = dataset['paraphrased_predictions'].apply(lambda data: reformat(literal_eval(data)))

    # extract all the predictions
    dataset['normal_prediction_tokens'] = dataset['normal_predictions'].apply(lambda data: extractPredictedTokens(data))
    dataset['paraphrased_prediction_tokens'] = dataset['paraphrased_predictions'].apply(lambda data: extractPredictedTokens(data))

    # score the matches
    dataset['normal_scores'] = dataset.apply(lambda page: scoreMatches(page['normal_prediction_tokens'], literal_eval(page['normal_entities'])), axis=1)
    dataset['paraphrased_scores'] = dataset.apply(lambda page: scoreMatches(page['paraphrased_prediction_tokens'], literal_eval(page['paraphrased_entities'])), axis=1)

    # small info about results
    normal_accuracy = dataset['normal_scores'].mean()
    paraphrased_accuracy = dataset['paraphrased_scores'].mean()
    print("Average Precision Scores:\nNormal: {}\nParaphrased: {}\n\n".format(normal_accuracy, paraphrased_accuracy))

    # # save results
    print("saving dataset")
    dataset.to_csv(resultsFile, index=False)
