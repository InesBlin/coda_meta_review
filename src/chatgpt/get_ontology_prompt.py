# -*- coding: utf-8 -*-
"""
"""
from rdflib import Graph
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_RDF_XML
from src.helpers import rdflib_to_pd

SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
PRED_TO_ABBR = {
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'rdf:type',      'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'rdfs:subClassOf'
}


QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dct: <http://purl.org/dc/terms/>
CONSTRUCT {
    ?siv rdfs:subClassOf ?giv ;
         rdfs:label ?siv_l ;
         dct:description ?siv_d .
    ?sivv rdf:type ?siv ;
          rdfs:label ?sivv_l ;
          dct:description ?sivv_d .
    ?giv rdfs:label ?giv_l ;
         dct:description ?giv_d .
} WHERE { 
    ?sivv rdf:type ?siv ;
          rdfs:label ?sivv_l .
    OPTIONAL {?sivv dct:description ?sivv_d .}
	?siv rdfs:subClassOf cdo:IndependentVariable, ?giv ;
              rdfs:label ?siv_l .
    OPTIONAL {?siv dct:description ?siv_d .}
    ?giv rdfs:label ?giv_l .
    OPTIONAL {?giv dct:description ?giv_d .}
    FILTER(?giv IN (cdo:PunishmentVariable))
} 
"""

PROMPT_TEMPLATE = """
In this task, you need to generate triples based on an ontology. I first give you an example with a sample ontology, and then the real ontology.

[EXAMPLE]
I have the following triples in my knowledge graph:
(cdo:individualDifference, rdfs:subClassOf, cdo:IndependentVariable)
(cdo:individualDifference, rdfs:subClassOf, cdo:PersonalityVariable)
(cdo:individualism, rdf:type,cdo:individualDifference)
(cdo:honesty-humility, rdf:type, cdo:individualDifference)
(cdo:risk-taking, rdf:type,cdo:individualDifference)
(cdo:collectivism, rdf:type, cdo:individualDifference)

In this example, cdo:PersonalityVariable is a GIV, cdo:individualDifference is a SIVV and cod:individualism, cdo:honesty-humility, cdo:risk-taking, and cdo:collectivism are SIVV.

Based on this small ontology, can you generate hypotheses in the form:
```
Cooperation is significantly <higher OR lower> when <SIV1> is <SIVV1>, compared to when <SIV2> is <SIVV2>.
```
The hypotheses should be non redundant and sensible. SIV1 and SIV2 should be SIVs from the ontology, and SIVV1 and SIVV2 SIVVs from the ontology.

One example would be:
```
Cooperation is significantly higher when <cdo:individualDifference> is <cdo:honesty-humility>, compared to when <cdo:individualDifference> is <cdo:individualism>.
```

[REAL]
Now here are the triples from which you need to generate hypotheses
```
<TRIPLES>
```

Can you generate up to 5 sensible and coherent hypotheses? Remember, you need to keep the bracket and the original names in the ontology when you generate the hypotheses.
"""

def write_triples(filtered):
    res = []
    for _, row in filtered.iterrows():
        res.append(f"(cdo:{row.subject.split('/')[-1]}, {PRED_TO_ABBR[row.predicate]}, cdo:{row.object.split('/')[-1]})")
    return "\n".join(res)

def main():
    response = run_query(query=QUERY, sparql_endpoint=SPARQL_ENDPOINT, headers=HEADERS_RDF_XML)
    graph = Graph()
    graph.parse(data=response.text, format="application/rdf+xml")
    df = rdflib_to_pd(graph=graph)
    print(df.predicate.unique())

    filtered = df[df.predicate.isin(PRED_TO_ABBR.keys())].sort_values(by="predicate", ascending=False)
    ontology = write_triples(filtered=filtered)
    prompt = PROMPT_TEMPLATE.replace("<TRIPLES>", ontology)
    print(prompt)



if __name__ == '__main__':
    main()