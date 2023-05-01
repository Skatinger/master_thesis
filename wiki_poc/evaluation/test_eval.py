from eval import Evaluator
from datasets import load_dataset
gt = load_dataset("rcds/wikipedia-for-mask-filling", "original_512", split="train")
dataset = load_dataset("json", data_files="wiki_predictions_xlm-roberta-base_original_512.jsonl", split="train")
ev = Evaluator()
ev.init(dataset, gt, shard_size=1000)
ev.compute_accuracy_matrix()

# ev.top_k_accuracy(5)

ev.join_examples_by_page()

# import pdb; pdb.set_trace()
ev.most_frequent_prediction_accuracy(ev.by_page)
