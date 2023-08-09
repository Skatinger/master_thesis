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


# articles that re-identify:


René Peter Baumann (DJ Bobo)
4C.141/2002
https://www.nzz.ch/newzzD8FKELOE-12-ld.233470?reduced=true
https://www.nau.ch/people/aus-der-schweiz/was-leicht-illegales-so-kam-dj-bobo-auf-seinen-kunstlernamen-65909012
https://www.news.de/promis/855913087/dj-bobo-privat-frau-nancy-baumann-verheiratet-kinder-sohn-tochter-richtiger-name-rene-baumann-vater-sing-meinen-song-aktuell/1/

Max Beeler
9C_617/2011
https://www.humanrights.ch/de/anlaufstelle-strategische-prozessfuehrung/falldokumentation/diskriminierende-witwerrente/

Brian Keller
BGer 6B_356/2022
BGer 7B_188/2023
https://www.watson.ch/schweiz/z%C3%BCrich/524838509-bundesgericht-hat-entschieden-brians-psychiater-nicht-freigesprochen
https://www.zueritoday.ch/zuerich/bundesgericht-weist-urteil-gegen-aerzte-von-brian-zurueck-152503202
https://www.republik.ch/2023/07/28/die-unendliche-haftgeschichte-des-brian-keller
https://www.strafprozess.ch/punktsieg-fuer-brian/
https://www.republik.ch/2022/11/23/am-gericht-darum-soll-brian-keller-hinter-gittern-bleiben
https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?highlight_docid=aza://24-07-2023-7B_188-2023&lang=de&zoom=&type=show_document
https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?lang=de&type=highlight_simple_query&page=1&from_date=&to_date=&sort=relevance&insertion_date=&top_subcollection_aza=all&query_words=6B_356%2F2022&rank=1&azaclir=aza&highlight_docid=aza%3A%2F%2F23-06-2023-6B_356-2022&number_of_ranks=152



6B_851/2015
Beschwerdeführerin:  Kathy Rinklin
Beschwerdegegner: Christoph Mörgeli
https://www.nzz.ch/schweiz/bundesgericht-verurteilt-kathy-riklin-ld.9571?reduced=true


Patrick Stach
2C_205/2019
https://insideparadeplatz.ch/2020/01/14/st-galler-posterboy-anwalt-in-eisigem-wind/

Hans Kneubühl
Urteil Urteil 6B_455/2021 (https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?highlight_docid=aza%3A%2F%2F23-06-2021-6B_455-2021&lang=de&type=show_document&zoom=YES&)
News artikel ganz viele, nach kneubühl suchen


Peter Studler
https://www.aargauerzeitung.ch/aargau/lenzburg/zwei-neue-anzeigen-es-ist-etwas-faul-bei-backer-peter-studler-ld.1374847
https://www.blick.ch/schweiz/ex-angestellte-im-clinch-mit-aargauer-gastro-kette-lohn-gekuerzt-weil-zimmer-dreckig-waren-id15403266.html
https://www.aargauerzeitung.ch/aargau/lenzburg/lenzburglausanne-baecker-peter-studler-siegt-vor-bundesgericht-ld.2168825
https://www.aargauerzeitung.ch/aargau/lenzburg/streit-um-bauarbeiten-aargauer-unternehmer-von-gericht-wegen-notigung-verurteilt-ld.1372475
urteil (BG): https://servat.unibe.ch/dfr/bger/2022/220224_4A_57-2022.html
Urteil zur nötigung leider nicht gefunden

Paul Estermann
https://www.luzernerzeitung.ch/zentralschweiz/luzern/luzern-kantonsgericht-spricht-springreiter-paul-estermann-wegen-tierquaelerei-schuldig-ld.2087798
ID: 45701417
https://www.fnch.ch/de/Disziplinen/Springen/News-aus-der-Disziplin-2/Sieben-Jahre-Sperre-fuer-Springreiter-Paul-Estermann.html
https://www.luzernerzeitung.ch/thema/paul-estermann
Urteil 6B_576/2021
https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?lang=de&type=highlight_simple_similar_documents&page=81&from_date=01.01.2005&to_date=&sort=relevance&insertion_date=&top_subcollection_aza=all&docid=aza%3A%2F%2F15-08-2017-5D_141-2017&rank=802&azaclir=aza&highlight_docid=aza%3A%2F%2F21-02-2022-6B_576-2021&number_of_ranks=34410


Werner Fleischmann
https://www.luzernerzeitung.ch/zentralschweiz/luzern/spielsuechtiger-pfarrer-luzerner-strafbehoerden-muessen-ermitteln-ld.1129220
https://www.luzernerzeitung.ch/zentralschweiz/luzern/spielsuechtiger-kuessnachter-pfarrer-muss-gehen-ld.1030225
Den Beschluss des Bundesstrafgerichts finden Sie auf «bstger.weblaw.ch». Geschäftsnummer BG.2019.15.


Personen im Artikel bekannt, werden im Urteil aber nicht anonymsiert sondern als "zwei Privatkläger" genannt,
allerdings es ist auch ein Gerichtsstandskonflikt zwischen Zürich und Bern und kein Urteil.
https://entscheidsuche.ch/search?query=BG.2012.26&selected=CH_BSTG_001_BG-2012-26_2012-09-25&preview=true
artikel im bund.ch: ID: 2735666
Keine Immunität hätten Christoph Blocher, Walter Frey und Nadja Pieren, die im Sommer 2011 noch Berner Grossrätin war. Falls nicht nur die Co-Präsidenten des Komitees belangt würden, sondern auch dessen Mitglieder, wären der Zürcher Banker Thomas Matter sowie die damals kantonalen Parlamentarier Céline Amaudruz (GE), Anita Borer (ZH), Erich Hess (BE) und Pierre Rusconi (TI) ohne Immunitätsschutz. Dasselbe gilt für Generalsekretär Martin Baltisser, der als Mitglied der SVP-Führung ebenfalls ins Visier der Ermittlungen geraten könnte.
