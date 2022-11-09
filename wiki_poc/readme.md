# installation
## Local
1. `python -m venv .env`
2. `source .env/bin/activate`
3. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
4. `pip install -r requirements.txt`


## Ubelix




## Scripts for Wiki-POC
Use either the `.py` file locally or on ubelix with your desired options, or run the `.sh` files for default options.
Ensure you load python env and cuda when running without default options.


`download-wiki.py`: downloads the wikipedia dataset required for the downstream tasks, run this first  
`build-paraphrased`: uses the wiki dataset to build collections of paraphrased sentences  
`build-unparaphrased`: the same as build-paraphrased, but keeps the sentences  
`build-masked`: uses the sentences preprocessed in `build-*` and masks entities identified as the person on which 
the wikipedia article is about.

### Good to Know
Texts are split to sentences for paraphrasing only, as this will ensure the paraphrasing only does small changes on the
sentence structure, but not change the full text. Sentences are then joined back together to a text, to improve the
quality of NER, as well as the fill-mask afterwards.

### Fields of the datasets
`id`: the id of the data page within the wiki dataset  
`normal_text_masked`: contains the normal text once for every <mask>, so for 5 detected entities, it contains 5 texts with each entity masked once.  

## Scripts and Pipeline for Wiki-Large
1. `prepare-wiki-large`:
    pulls wikipedia dataset and first 700'000 people from wikipedia via sparql query on wikidata. Only keeps wikipedia pages
    which match a person from the 700'000 people found in the sparql query. Saved into `./data`
2. `clean-wiki-large`:
    removes bibliographies and unwanted sections, removes articles shorter than 6'000 characters. Processing might throw some
    "Token indices sequence too long", which is fine. It indicates that the sentence will be truncated. Saved to `./data_reduced`
3. `build-unparaphrased-large`:
    splits raw text of pages into sentences. Saved to `./data_unparaphrased`
4. `build-paraphrased-large`:
    paraphrases sentences to separate dataset column. Checkpoints at `.build_paraphrased_checkpoint`, final result saved to `./data_paraphrased`

