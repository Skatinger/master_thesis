# recognizes entities in the given sentences and masks them accordingly
# stores masked sentences and the tokens belonging to the masks


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_from_disk
import functools
import re

# sigint handler
import signal
import sys
import logging
logging.getLogger().setLevel(logging.INFO)


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    # global dataset
    # dataset.save_to_disk(datasetPath)
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
datasetPath = './testing_shard_2'  # './data_paraphrased'


@functools.lru_cache(maxsize=1)
def load_ner_pipeline(model_name="dslim/bert-base-NER"):
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)


def masking(ner_results, text, entity_to_mask, batch_sizes, mask_token='<mask>'):
    person_nr = -1
    entities = []
    indexes = []
    index_start = 0
    for batch_size, result in zip(batch_sizes, ner_results):
        # reset last_end index for each batch, we do not want to concat entities from different batches, as batches
        # are separated by sentences (and thus entities cannot overlap)
        last_end = -10
        for entity in result:
            tag = entity['entity']
            if 'PER' in tag:  # if it is a person
                # import pdb; pdb.set_trace()
                if ('B' in tag or 'I' in tag) and '#' not in entity['word']:
                    # if end of entity is right after the end of the last entity, merge them
                    if entity['start'] - 1 == last_end:
                        # concat to last entity part with empty space between
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        person_nr += 1  # we came to the next person
                        entities.append(entity['word'].strip())
                        indexes.append([entity['start'] + index_start, entity['end'] + index_start])
                        last_end = entity['end']
                else:
                    if person_nr < 0:
                        continue
                    if '#' in entity['word']:
                        entities[person_nr] += entity['word'].strip().strip('#')
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
        # update index_start for next batch, so indexes can be directly used for masking in full text
        # otherwise we would have to add the length of the previous batch to the indexes
        index_start += batch_size
    # remove entities which are not the ones we want to mask, e.g. remove persons which
    # are not the person the article is about
    # complex checker:
    regex = ".*".join(entity_to_mask.split(" "))
    regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity_to_mask.split(" ")])

    remaining_entities = []
    remaining_indexes = []
    for entity, index in zip(entities, indexes):
        if bool(re.match(regex, entity)):
            remaining_entities.append(entity)
            remaining_indexes.append(index)

    # return a dataset of anonymized strings with the belonging entities
    # https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
    # escape entity to ensure no brackets or other regex keywords are present (would cause a parsing error), for example
    # 'Heinrich ) Frank' causes a unbalanced parenthesis error
    text = "".join(text)
    for index in reversed(remaining_indexes):
        # the index start of an entity changes when text before it is masked, so we mask in reverse order
        text = text[:index[0]] + mask_token + text[index[1]:]

    return text, remaining_entities
    # return [re.sub(entity, mask_token, text) for entity in remaining_entities], remaining_entities


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at ./data not available")
        logging.warning(err)
        quit()


def prepare_paraphrased(example):
    return prepare_batched(example, 'paraphrased', 'paraphrased_sentences')


def prepare_original(example):
    return prepare_batched(example, 'original', 'sentences')


# runs the ner pipeline on the given example and masks the entities
# example is a wiki page, type is either 'sentences' or 'paraphrased_sentences', depending on which column to use
def prepare_batched(example, type, colname):
    # split sentences array in batches of splitSize 7
    splitSize = 7
    # [["sentence 1", "sentence 2", ...], [...]]
    sentencesBatches = [example[colname][i:i + splitSize] for i in range(0, len(example[colname]), splitSize)]

    # join sentences within each batch with spaces to get a single string to pass to NER pipeline
    sentencesBatches = [' '.join(sentences) for sentences in sentencesBatches]
    batchesLenths = [len(text) for text in sentencesBatches]
    return {'batched_{}_sentences'.format(type): sentencesBatches, 'batched_{}_sizes'.format(type): batchesLenths}


def apply_original_ner(example):
    return apply_ner(example, 'original')


def apply_paraphrased_ner(example):
    return apply_ner(example, 'paraphrased')


# performs ner on the given example on the given column
def apply_ner(example, type):
    ner = load_ner_pipeline()
    column = 'batched_{}_sentences'.format(type)
    ner_results = ner(example[column])
    return {"ner_results_{}".format(type): ner_results}


def apply_original_masking(example):
    return apply_masking(example, 'original')


def apply_paraphrased_masking(example):
    return apply_masking(example, 'paraphrased')


# masks the entities in the given example on the given column
def apply_masking(example, type):
    # mask entities in the text
    ner_results = example['ner_results_{}'.format(type)]
    batched_text = example['batched_{}_sentences'.format(type)]
    batch_sizes = example['batched_{}_sizes'.format(type)]
    masked_text, entities = masking(ner_results, batched_text, example['title'], batch_sizes)

    return {'masked_text_{}'.format(type): masked_text, 'masked_{}_entities'.format(type): entities}


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()

    # split sentences of each page into batches of 7 sentences, and sentences
    # within each batch into a single string for better performance and accuracy
    logging.info('Splitting sentences into batches...')
    dataset = dataset.map(prepare_original, num_proc=1)
    dataset = dataset.map(prepare_paraphrased, num_proc=1)

    # apply ner pipeline to each batch of sentences
    logging.info('Applying NER pipeline...')
    dataset = dataset.map(apply_original_ner, num_proc=1)
    dataset = dataset.map(apply_paraphrased_ner, num_proc=1)

    # mask entities detected in NER
    logging.info('Masking entities...')
    dataset = dataset.map(apply_original_masking, num_proc=1)
    dataset = dataset.map(apply_paraphrased_masking, num_proc=1)

    # drop rows where we got no entities, as they are not interesting for our project
    logging.info('Dropping rows without entities...')
    dataset = dataset.filter(lambda example: len(example['masked_original_entities']) > 0)
    dataset = dataset.filter(lambda example: len(example['masked_paraphrased_entities']) > 0)

    # remove columns we don't need anymore
    logging.info('Removing unnecessary columns...')
    cols = ['batched_original_sentences', 'batched_original_sizes', 'batched_paraphrased_sentences',
            'batched_paraphrased_sizes', 'ner_results_original', 'ner_results_paraphrased']
    dataset = dataset.remove_columns(cols)

    # save the dataset
    logging.info('Saving dataset to ./data_masked...')
    dataset.save_to_disk('./data_masked')
