# recognizes entities in the given sentences and masks them accordingly
# stores masked sentences and the tokens belonging to the masks


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import logging
from ast import literal_eval
import os
import functools
import pandas as pd
import re
from tqdm import tqdm

# sigint handler
import signal
import sys


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.to_csv(dataset_file, index=False)
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
dataset_file = 'wiki-dataset.csv'


@functools.lru_cache(maxsize=1)
def load_ner_pipeline(model_name="dslim/bert-base-NER"):
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)


def masking(ner_results, text, entity_to_mask, mask_token='<mask>'):
    person_nr = -1
    entities = []
    for entity in ner_results:
        tag = entity['entity']
        if 'PER' in tag:  # if it is a person
            if 'B' in tag and '#' not in entity['word']:
                person_nr += 1  # we came to the next person
                entities.append(entity['word'].strip())
            else:
                if '#' in entity['word']:
                    entities[person_nr] += entity['word'].strip().strip('#')
                else:
                    entities[person_nr] += ' ' + entity['word']

    # remove entities which are not the ones we want to mask, e.g. remove persons which are not the person the article is about
    # complex checker:
    regex = ".*".join(entity_to_mask.split(" "))
    regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity_to_mask.split(" ")])

    remaining_entities = []
    for entity in entities:
        if bool(re.match(regex, entity)):
            remaining_entities.append(entity)

    # return a dataset of anonymized strings with the belonging entities
    # https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
    # escape entity to ensure no brackets or other regex keywords are present (would cause a parsing error), for example
    # 'Heinrich ) Frank' causes a unbalanced parenthesis error
    for entity in remaining_entities:
        text = re.sub(re.escape(entity), mask_token, text)

    return text, remaining_entities
    # return [re.sub(entity, mask_token, text) for entity in remaining_entities], remaining_entities


if __name__ == '__main__':
    if not os.path.exists(dataset_file):
        logging.error("Input file does not exist. Please run `download-wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(dataset_file)
    if 'sentences' not in dataset.columns:
        logging.error("Input file does not contain a 'sentences' column. Please run `build-unparaphrased.py` first.")
        quit()
    if 'paraphrased_sentences' not in dataset.columns:
        logging.error("Input file does not contain a 'paraphrased_sentences' column. Please run `build-paraphrased.py` first.")
        quit()

    # parse sentences string back to list of sentences
    dataset['sentences'] = dataset['sentences'].apply(literal_eval)
    dataset['paraphrased_sentences'] = dataset['paraphrased_sentences'].apply(literal_eval)

    # convert sentences to text
    dataset['normal_text'] = dataset['sentences'].apply(lambda x: " ".join(x))
    dataset['paraphrased_text'] = dataset['paraphrased_sentences'].apply(lambda x: " ".join(x))

    # add dataset columns for masking results if not yet existing
    if 'normal_masked_text' not in dataset.columns:
        dataset['normal_masked_text'] = ""
        dataset['normal_entities'] = ""
        dataset['paraphrased_masked_text'] = ""
        dataset['paraphrased_entities'] = ""

    # perform NER on the texts
    ner = load_ner_pipeline()
    tqdm.pandas()
    print("Apply NER to normal text...")

    for index, row in dataset.iterrows():
        # skip iteration if value already present
        if (isinstance(row['normal_masked_text'], str)):
            logging.info("Skipping page {}, already done.".format(index))
            continue

        # normal text
        ner_result_unparaphrased = ner(row['normal_text'])
        dataset.at[index, 'normal_masked_text'], dataset.at[index, 'normal_entities'] = masking(ner_result_unparaphrased, row['normal_text'], row['title'])

        # paraphrased text
        ner_result_paraphrased = ner(row['paraphrased_text'])
        dataset.at[index, 'paraphrased_text'], dataset.at[index, 'normal_entities'] = masking(ner_result_paraphrased, row['paraphrased_text'], row['title'])

        if (index % 5 == 0):
            logging.info("Checkpointing at page {}".format(index))
            dataset.to_csv(dataset_file, index=False)

    # drop rows where we got no entities, as they are not interesting for our project
    cleanedDataset = dataset[dataset['normal_entities'].apply(lambda x: len(x) > 0)]
    cleanedDataset = dataset[dataset['paraphrased_entities'].apply(lambda x: len(x) > 0)]

    # save the dataset
    cleanedDataset.to_csv('wiki-dataset-masked.csv', index=False)
