

from transformers import pipeline
import timeit

inputs = ["Hello, I am a text with a <mask>", "Wow, there is <mask> more text.",
          "The largest building in <mask> is the taj mahal.", "Let's see if <mask> can win the f1 championship.",
          "<mask> are the only animals to live in these rough conditions.", "Not many people can say the have seen <mask>."]


fill_mask = pipeline("fill-mask", model="roberta-base", tokenizer='roberta-base', top_k=5)


d1 = timeit.timeit(lambda: [fill_mask(input) for input in inputs], number=1000)

d2 = timeit.timeit(lambda: fill_mask(inputs), number=1000)

print("Separated inputs took {}".format(d1))
print("Joined inputs took    {}".format(d2))
