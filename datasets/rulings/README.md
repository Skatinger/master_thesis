# SwissCourtDecisionReIdentification

In this work, we perform very specific re-identifications on Swiss court decisions using external data from SIMAP and STSB.

# Tipps
## Extracting dump of court decisions from postgres db of SwissCourtRulingsCorpus
1. log in to psql shell with `psql -d scrc -U readonly -W`
2. run copy command for all languages and pipe to desired file:
`\copy (SELECT id, spider, language, canton, court, chamber, date, file_name, file_number, pdf_url, text, header, title, judges, parties, topic, facts, considerations, rulings, footer, num_tokens FROM fr) to '/home/alex/de_04042023.csv' WITH CSV HEADER;`