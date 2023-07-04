# Observations

- it seems that mask filling performs better for lower sized models and models
not fine tuned for single-shot and instructions. Models tuned for instructions are quickly pretty good
at understanding the task, and due to the ability to ingest larger input and being trained on more data,
the results are better with instruction tuned models on with text generation.