# -*- coding: utf-8 -*-
"""
SPARQL queries lab
"""

SIMPLE_COUNTRY_MOD_QUERY = """
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?country ?processedLabel ?value WHERE {
    ?country rdf:type cdo:Country ;
       ?p ?value .
    ?p rdfs:label ?pLabel.
    FILTER ( ?p != rdfs:label).
    FILTER ( ?p != rdf:type).
    FILTER (!CONTAINS(str(?p), "https://data.cooperationdatabank.org/")) .
    BIND(REPLACE(LCASE(?pLabel), " ", "_") AS ?processedLabel)
}
"""

COMPLEX_COUNTRY_MOD_QUERY  = """
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
SELECT DISTINCT ?country (STR(?Year) as ?year) ?processedLabel ?value WHERE {
    ?country rdf:type cdo:Country ;
       ?p ?bn .
    ?p rdfs:label ?pLabel.
    ?bn cdp:year ?Year ;
        cdp:value ?value .
    FILTER ( ?p != rdfs:label).
    FILTER ( ?p != rdf:type).
    FILTER (CONTAINS(str(?p), "https://data.cooperationdatabank.org/")) .
    BIND(REPLACE(LCASE(?pLabel), " ", "_") AS ?processedLabel)
}
"""

VARIABLE_MOD_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
SELECT ?giv ?siv (LCASE(?label) AS ?siv_label)
WHERE {
    ?siv rdfs:subPropertyOf ?giv ;
    	 rdfs:label ?label .
    FILTER(CONTAINS(STR(?giv), "Variable"))
    FILTER(?giv != ?siv)
}
ORDER BY ASC(?siv)
"""