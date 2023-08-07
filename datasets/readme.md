# Datasets for the project

## News Articles
Files should be placed in datasets/news_articles, however they are not publicly available as per
the usage agreements this is not permitted.


Queried from Swissdox, with the following parameters:  
```yml
query:
  dates:
    - from: 2019-01-01
      to: 2019-12-31
  languages:
    - fr
    - de
    - it
  content:
    - verhandlung
    - urteil
    - gericht
    - schuldig
    - verurteilt
    - BGer
    - Freispruch
    - Anklage
    - Hinweis
    - Nötigung
    - Verfahren
    - angeklagt
    - Entscheid
    - Bundesgericht
    - Strafe
    - négociation
    - jugement
    - tribunal
    - coupable
    - condamné
    - BGer
    - acquittement
    - accusation
    - indice
    - contrainte
    - procédure
    - accusé
    - décision
    - Cour fédérale
    - peine
    - negoziazione
    - giudizio
    - tribunale
    - colpevole
    - condannato
    - BGer
    - assoluzione
    - accusa
    - indizio
    - coercizione
    - procedura
    - accusato
    - decisione
    - Corte federale
    - pena
result:
  format: TSV
  maxResults: 1020000
  columns:
    - id
    - pubtime
    - medium_code
    - medium_name
    - rubric
    - regional
    - doctype
    - doctype_description
    - language
    - char_count
    - dateline
    - head
    - subhead
    - content_id
    - content
version: 1.2
```

### Extracted decisions in news

| ID | Regex              | Filename         | Number of Results | Indication Format                        |
|----|--------------------|------------------|-------------------|------------------------------------------|
| 1  | `BG[0-9]{4}\.`     | by_regex_1.tsv   | 4                 | Urteil BG120322 vom 1. 10. 13            |
| 2  | `BG\.[0-9]{4}\.`   | by_regex_2.tsv   | 41                | Entscheid BG.2015.28 vom 2. 12. 15       |
| 3  | `Urteil\ [0-9]`    | by_regex_3.tsv   | 1892              | (Urteil 1C_6/2017 vom 25.10.2017)        |
| 4  | `Hinweis: Urteil`  | by_regex_4.tsv   | 18                | Hinweis: Urteil CA.2021.10               |


## How are relevant ones selected?
- using the decision_classification_helper.py news articles can be categorized manually, to check if they
  could be helpful to re-identify any person in a court ruling.
- if interesting articles are found, they are stored, and then the corresponding rulings are retrieved.
- The rulings can be retrieved by using the given ruling number in the news article.
- Other decisions identified manually by chance might be included, no specific process was involved,
  they were just found by pure chance.