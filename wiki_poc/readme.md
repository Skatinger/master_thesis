# installation
## Local
1. `python -m venv .env`
2. `source .env/bin/activate`
3. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
4. `pip install -r requirements.txt`


## Ubelix




## Scripts
Use either the `.py` file locally or on ubelix with your desired options, or run the `.sh` files for default options.
Ensure you load python env and cuda when running without default options.


`download-wiki.py`: downloads the wikipedia dataset required for the downstream tasks, run this first  
`build-paraphrased`: uses the wiki dataset to build collections of paraphrased sentences  
`build-unparaphrased`: the same as build-paraphrased, but keeps the sentences  
`build-masked`: uses the sentences preprocessed in `build-*` and masks entities identified as the person on which 
the wikipedia article is about.

## Good to Know
Texts are split to sentences for paraphrasing only, as this will ensure the paraphrasing only does small changes on the
sentence structure, but not change the full text. Sentences are then joined back together to a text, to improve the
quality of NER, as well as the fill-mask afterwards.