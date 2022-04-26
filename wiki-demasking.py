from mmap import PAGESIZE
# DOC
# 1) retrieves a file with wikipedia articles
# 2) tokenizes the articles and paraphrases them
# 3) performs named entity recognition on the paraphrased articles
# 4) masks the recognized person entities
# 5) predicts the masked entities with a BERT model
# 6) computes the accuracy of the predictions as a percentage of correctly predicted masks

# required python packages:
# - transformers
# - torch
# - pandas
# - re
# - sentencepiece
# - csv
# - nltk
# nltk.download('punkt')

## instantiate transformer model
from transformers import pipeline
print("Loading Fill-Mask model")
fill_mask = pipeline("fill-mask", model="roberta-base", tokenizer='roberta-base', top_k=5)
mask_token = fill_mask.tokenizer.mask_token

## use pegasus model to paraphrase text, to fit better within the maximum
## length constraint of BERT while keeping the most important information
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_model(model_name='tuner007/pegasus_paraphrase'):
    print(f"Loading {model_name}")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def paraphrase(input_texts, num_return_sequences=1, num_beams=10, max_length=60, temperature=1.5):
    model, tokenizer = load_model()
    print("paraphrasing...")
    batch = tokenizer(input_texts,
                      truncation=True, padding='longest', max_length=max_length, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=max_length, num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=temperature)
    paraphrased_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return paraphrased_texts

## use a tokenizer to split the wikipedia text into sentences
## Use a entity-recognition model to quickly find entities instead of labelling them by hand
# this helps to mask the entities, making it easier to work automatically instead of doing it manually
# can consider doing it by hand later on for better precision
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def load_ner_pipeline(model_name ="dslim/bert-base-NER"):
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)

def prepare_for_reidentification(sentences, mask_token):
    text = " ".join(sentences)
    print("loading NER pipeline")
    ner = load_ner_pipeline()
    print("performing NER")
    ner_results = ner(text)

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

    # return a dataset of anonymized strings with the belonging entities
    # the re.sub ## is required to concat BERT word splits back together
    # https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
    return [re.sub(entity, mask_token, text) for entity in entities], entities # [re.sub(" ##", '', entity) for entity in entities]

## Helper to predict masked entities and compute accuracy
def evaluate(texts, entities):
    assert len(texts) == len(entities)
    predictions = []
    for i in range(len(entities)):  # for all texts and entities
        text, entity = texts[i], entities[i]
        try:
          results = fill_mask(text)
        except:
          print("Got a sentence without token!")
          continue

        # results should be List[List[Dict]], but sometimes it is List[Dict]
        if not any(isinstance(el, list) for el in results):
            results = [results]  # transform it

        top_prediction = { 'predicted': results[0][0]['token_str'], 'actual': entity }
        predictions.append(top_prediction)

    # count how many times the prediction was in the target
    correct_preds_count = sum(f['predicted'] in f['actual'] or f['actual'] in f['predicted'] for f in predictions)
    if(len(predictions) > 0 and correct_preds_count > 0):
      print("got predicitions:")
      print(predictions)
      accuracy = 100 / len(predictions) * correct_preds_count
    else:
      print("got 0 accuracy for predictions:")
      print(predictions)
      accuracy = 0.0
    return accuracy

# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
def split_to_sentences(text):
  return sent_tokenize(text, 'english')

######## Main execution

## Import Data from CSV
import pandas as pd
file = 'wiki-dataset-top50.csv' # '/content/drive/MyDrive/wiki-dataset-reduced.csv'

assert len(file) > 0, "Please provide a file path to the dataset"
print("loading dataset")
dataset = pd.read_csv(file)

from nltk.tokenize import sent_tokenize

total_results = {}
# iterate over all the pages
for index, page in dataset.iterrows():
  # for fast debugging, only do one loop
  # if index > 8:
  #  break
  print("Now processing page:")
  print(page)
  # convert text to sentences, use only the first 1024 characters for performance reasons
  text = page['text'][:2056]
  print("STEP 1/4: splitting text into sentences")
  sentences = split_to_sentences(text)
  print("STEP 2/4: paraphrasing sentences")
  print("sentences:")
  print(sentences)
  paraphrased_sentences = paraphrase(sentences)
  print("paraphrased:")
  print(paraphrased_sentences)

  print("STEP 3/4: preparing for reidentification")
  anonymized, entities = prepare_for_reidentification(paraphrased_sentences, mask_token)
  print("anonymized:")
  print(anonymized)
  print("entities:")
  print(entities)

  # reidentify the entities
  print("STEP 4/4: Predicting masked tokens and computing accuracy")
  accuracy = evaluate(anonymized, entities)

  total_results[page['title']] = accuracy

  print(f"Approximate Accuracy for {page['title']}: {accuracy}%")

print(total_results)

