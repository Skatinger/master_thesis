# helpers to download, process and use wiki data

from SPARQLWrapper import SPARQLWrapper, JSON


# finds the given count of people from the english wikipedia
# returns a list of their names
def query_wiki_persons(count=10):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # find pages with english title from uk or en
    sparql.setQuery("""
    SELECT DISTINCT ?item ?page_titleEN ?articleLabel WHERE {
    ?item wdt:P31 wd:Q5.
    ?article schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/>;
        schema:name ?page_titleEN.
    ?item rdfs:label ?LabelEN.
    FILTER((LANG(?LabelEN)) = "en")
    ?item rdfs:label ?LabelUK.
    FILTER((LANG(?LabelUK)) = "uk")
    }
    LIMIT %d
    """ % (count))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # get the page titles of the queries persons
    return [page['page_titleEN']['value'] for page in results['results']['bindings']]
