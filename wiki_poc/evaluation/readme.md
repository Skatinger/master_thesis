# Evaluation Wiki POC

## Metrics
### Definition of "correct prediction"
It is more or less impossible to predict a name of a person exactly within a text, especially in wikipedia texts,
as the fullname, last, first and sometimes even middle names are used interchangeably.
Furthermore the title of the wikipedia page might not be the full name, or just a synonym, which makes predictions even harder.
Examples for this might be the wikipedia page title *John F. Kennedy* which refers to the full name *John Fitzgerald Kennedy*
within the text.
Artists are regulary referred to by their artist name in the title, but usually also in the text which is not a problem.
Only names which fit the title of the wikipedia page are masked. However, this might lead to relatives of the wikipedia
person to be masked as well. For example the wife of JFK might be masked as well as JFK himself, as they are both referenced
by the name Kennedy.

The masking did not differentiate between first and lastname, and therefore the evaluation will neither.
A prediction is therefore correct if part of the name (first, last or middle name) is correctly predicted.

### Computation of *best* prediction
How well a model can identify depends on which person it suggests first. Ranking predictions will be done as follows:

Any occurences of *correct predictions* (see definition above) within the top 5 predictions for each mask are tokenized together.
All other words are tokenized by their case insensitive string. For each token group, the prediction scores of the group is summed up.
This gives a ranking of predicted tokens by their overall prediction score. The token with the best score is most likely the
title of the wikipedia page. Simplified example:

**Sentence:** *`<mask>` was the president of the United States in the years 2014 - 2018. `<mask>` was born in Nebraska. With his wife `<mask>`
moved to Boston to follow his political career.*

**Predictions (5 per mask) with their scores::**
1. John Fitzgerald Kennedy, Kennedy, John F. Kennedy, He ==> [0.45, 0.12, 0.11, 0.15, 0.12]
2. He, Kennedy, Trumann, His, Junior ==> [0.62, 0.22, 0.21, 0.12, 0.05]
3. Margareth Kennedy, Kennedy, together, he [0.48, 0.44, 0.31, 0.23, 0.06]

With all *correct* predictions replaced with the same token we get:
1. John F. Kennedy, John F. Kennedy, John F. Kennedy, He ==> [0.45, 0.12, 0.11, 0.15, 0.12]
2. He, John F. Kennedy, Trumann, His, Junior ==> [0.62, 0.22, 0.21, 0.12, 0.05]
3. John F. Kennedy, John F. Kennedy, together, he [0.48, 0.44, 0.31, 0.23, 0.06]

Now summing up all scores of names gives:
| id | Title           | Score | Calculation                                   |
|----|-----------------|-------|-----------------------------------------------|
| 1  | John F. Kennedy | 1.76  | (0.45 + 0.12 + 0.11) + (0.22) + (0.48 + 0.44) |
| 2  | he              | 0.8   | (0.12) + (0.62) + (0.06)                      |
| 3  | together        | 0.23  | () + () + (0.23)                              |

Which makes John F. Kennedy the best and also the correct prediction. This will however not always be the case.

### Mask specific performance
Measures how good the predictions for any given mask are. This is the toughest metric, as chances are that most wikipedia
pages will be identified 

### Example specific performance
Checks if the top prediction for a full chunk of a page is correct. This is done with the computation explained below.

### per page performance



### Other comparisions / measures
- How much better would the models perform if all occurences of the entity were not masked, just the mask which is
being predicted? e.g. it could read the name coming from different places in the text.

### Runnables
python -m evaluation.top_k_evaluation <key-of-run>
python -m evaluation.plotting.precomputed_plotting <key-of-run>