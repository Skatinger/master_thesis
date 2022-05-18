## uses the raw wiki-dataset to build a csv with paraphrased sentences as wiki text
## uses the model tuner007/pegasus_paraphrase to paraphrase the wiki-dataset
## runs on CUDA if possible

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

input_filepath = 'csv/wiki-dataset-raw.csv'
output_filepath = 'csv/wiki-dataset-paraphrased.csv'
sizelimit = 1024 #4096 # maximum text length before paraphrasing

# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
# def split_to_sentences(text):
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         logging.info("Downloading nltk punkt tokenizer")
#         nltk.download('punkt')
#     return nltk.tokenize.sent_tokenize(text, 'english')

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)

# ensure model only once
@functools.lru_cache(maxsize=None)
def load_model(model_name='tuner007/pegasus_paraphrase'):
    print(f"Loading {model_name}")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def paraphrase(input_texts, num_return_sequences=1, num_beams=10, max_length=60, temperature=1.5):
    model, tokenizer = load_model()
    batch = tokenizer(input_texts,
                      truncation = True,
                      padding = 'longest',
                      max_length = max_length,
                      return_tensors = "pt"
                      ).to(torch_device)
    translated = model.generate(**batch, max_length=max_length, num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=temperature)
    paraphrased_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return paraphrased_texts


if __name__ == '__main__':
    if os.path.exists(output_filepath):
        logging.warning("Output file already exists, skipping `build-paraphrased.py`")
        quit()
    if not os.path.exists(input_filepath):
        logging.error("Input file does not exist. Please run `download_wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(input_filepath)

    # paraphrase the wiki-dataset
    for index, page in dataset.iterrows():
        paraphrased_texts = paraphrase(dataset['text'].tolist())

    dataset['paraphrased_sentences'] = paraphrased_texts
    dataset.drop(columns=['text'], inplace=True)

    save_to_csv(dataset, output_filepath)
    logging.info("Saved unparaphrased wiki-dataset to {}".format(output_filepath))
