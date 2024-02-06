# -*- coding: utf-8 -*-
"""
SPARQL queries lab
"""

QUERY_DIFFERENCE_COMPARED_TREATMENT = """
PREFIX coda: <https://data.cooperationdatabank.org/>
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?p1 ?o1 ?o2 (COUNT(DISTINCT(?obs)) as ?nb_obs) WHERE { 
	?obs rdf:type cdo:Observation ;
      	 cdp:treatment ?t1, ?t2 .
    FILTER(?t1 != ?t2)
    FILTER(str(?t1) < str(?t2))
    
    ?t1 ?p1 ?o1 .
    ?t2 ?p2 ?o2 .
    
    FILTER(?p1 = ?p2)
    FILTER(!CONTAINS(STR(?p1), "Variable"))
    FILTER(!CONTAINS(STR(?p1), "rdf"))
    FILTER(!CONTAINS(STR(?p2), "rdf"))
    FILTER(?o1 != ?o2)
    FILTER(str(?o1) < str(?o2))
    FILTER(CONTAINS(STR(?o1), "http"))
    FILTER(CONTAINS(STR(?o2), "http"))
}
GROUP BY ?p1 ?o1 ?o
ORDER BY DESC(?nb_obs)
"""


QUERY_DIFFERENCE_COMPARED_TREATMENT_TEMPLATE = """
PREFIX coda: <https://data.cooperationdatabank.org/>
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
select distinct ?country ?obs ?e ?t1 ?t2 ?dilemma where { 
    ?study cdp:reportsEffect ?obs ;
           cdp:country ?country ;
           cdp:studyDilemmaType ?dilemma .
	?obs rdf:type cdo:Observation ;
      	 cdp:eSEstimate ?e ;
      	 cdp:treatment ?t1, ?t2 .
    ?t1 <predicate-to-replace> <obj1-to-replace> .
    ?t2 <predicate-to-replace> <obj2-to-replace> .
}
"""