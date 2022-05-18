## uses the raw wiki-dataset to build a csv with paraphrased sentences as wiki text
## uses the model tuner007/pegasus_paraphrase to paraphrase the wiki-dataset
## runs on CUDA if possible
## paraphrases each sentence seperately, to keep the length of the text, and only scramble the words a bit

from ast import literal_eval
import faulthandler
import functools
import nltk
import pandas as pd
import logging
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# ensure error stack is printed when an error occurs on the GPU / Computing Cluster
faulthandler.enable()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_file = 'wiki-dataset.csv'
sizelimit = 4096 # maximum text length before paraphrasing

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)

# ensure model is loaded only once
@functools.lru_cache(maxsize=1)
def load_model(model_name='tuner007/pegasus_paraphrase'):
    print(f"Loading {model_name}")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def paraphrase_sentence(input_texts, num_return_sequences=1, num_beams=10, temperature=1.5):
    model, tokenizer = load_model()
    batch = tokenizer(input_texts,
                      truncation = True,
                      padding = 'longest',
                      return_tensors = "pt"
                      ).to(torch_device)
    
    translated = model.generate(**batch,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=temperature)
    
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

if __name__ == '__main__':
    if not os.path.exists(dataset_file):
        logging.error("Input file does not exist. Please run `download-wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(dataset_file)
    if not 'sentences' in dataset.columns:
        logging.error("Input file does not contain a 'sentences' column. Please run `build-unparaphrased.py` first.")
        quit()

    # parse sentences string back to list of sentences
    dataset['sentences'] = dataset['sentences'].apply(literal_eval)

    # paraphrase the wiki-dataset
    paraphrased_texts = []

    # iterate over all wiki pages
    for index, page in dataset.iterrows():
        print("Processing page #" + str(index))
        
        paraphrase_sentences = []
        # iterate over all sentences in the wiki-page
        nb_pages = len(page['sentences'])
        for sindex, sentence in enumerate(page['sentences']):
            print("processing sentence " + str(sindex) + "/" + str(nb_pages))
            paraphrase_sentences.append(paraphrase_sentence(sentence))
        
        # create a list of paraphrased sentences for each wiki-page
        paraphrased_texts.append(paraphrase_sentences)

    dataset['paraphrased_sentences'] = paraphrased_texts

    save_to_csv(dataset, dataset_file)
    logging.info("Saved unparaphrased wiki-dataset to {}".format(dataset_file))
