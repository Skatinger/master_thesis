# Datasets for the project

## News Articles
Files should be placed in datasets/news_articles, however they are not publicly available as per
the usage agreements this is not permitted.


Queried from Swissdox, with the following parameters:  
```yml
query:
  sources:
    - ZWA
    - AZ
    - AT
    - AZM
    - APPZ
    - BLZ
    - BAZ
    - BEO
    - BEOL
    - BZ
    - BIZ
    - BU
    - FUW
    - FUWM
    - SHZ
    - LUZ
    - MLZ
    - NEWS
    - CAMP
    - NZZG
    - NZZS
    - NZZ
    - NIW
    - SGT
    - AZO
    - ZUE
    - NNBE
    - NNBU
    - SHZO
    - NZZO
    - SRF
  dateRange:
    - 2012-04-01..2022-04-01
  languages:
    - de # TODO: Add french news articles as well
  doctypes:
    - PLD
    - PLW
    - WWE
    - PRD
    - PJO
    - PND
    - NNE
    - PMA
  content:
    - verhandlung
    - urteil
    - verurteilt
    - bundesgericht
    - gericht
    - strafe
    - schuldig
    - BGer
    - Entscheid
    - freispruch
    - spricht frei
    - anklage
    - angeklagt
    - Verfahren
    - Nötigung
    - Hinweis
result:
  format: TSV
  maxResults: 10000000
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
version: 1.1
```

### Extracted decisions in news

| id | regex | filename | number of results |  indication format |
|----|-------|----------|-------------------|---|
|  1  |   `BG[0-9]{4}\.`    |    by_regex_1.tsv      |        4           |  Urteil BG120322 vom 1. 10. 13 |
|  2  |   `BG\.[0-9]{4}\.`    |    by_regex_2.tsv    |        41          |  Entscheid BG.2015.28 vom 2. 12. 15 |
|  3  |    `Urteil\ [0-9]`   |     by_regex_3.tsv    |        1892        | (Urteil 1C_6/2017 vom 25.10.2017)  |
|  4  |   `Hinweis: Urteil`  |     by_regex_4.tsv    |        18          |  Hinweis: Urteil CA.2021.10  |
|  5  |       |          |                   |   |
|  6  |       |          |                   |   |
