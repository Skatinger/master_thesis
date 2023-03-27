
from datasets import load_dataset


"""
creates a file models/test_set_ids.csv with the ids of the test set of the wikipedia dataset
ids are sharded from the base dataset, identifying 10'000 pages.
"""

base = load_dataset("Skatinger/wikipedia-persons-masked", split="train")
factor = len(base) / 10000
shard = base.shard(factor, 0)

with open("models/test_set_ids.csv", "w") as f:
    for i in shard:
        f.write(i["id"] + "\n")
