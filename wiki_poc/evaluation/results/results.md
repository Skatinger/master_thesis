# Observations

- it seems that mask filling performs better for lower sized models and models
not fine tuned for single-shot and instructions. Models tuned for instructions are quickly pretty good
at understanding the task, and due to the ability to ingest larger input and being trained on more data,
the results are better with instruction tuned models on with text generation.

- roberta_squad performed worse than roberta, but this is most likely due to the fact that it is harder
to instruct models what to do, especially with such specific tasks as re identifying a person in a text.
The same sized model not tuned for instruction was a mask-filling model, so the task was already clear without
any fine-tuning and therefore the performance was better.