
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

def main():

    rulings = load_dataset("rcds/swiss_rulings", split="train")
    

    tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
    model = AutoModel.from_pretrained("ZurichNLP/swissbert")
    model.set_default_language("de_CH")

if __name__ == "__main__":
    main()