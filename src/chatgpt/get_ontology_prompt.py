# -*- coding: utf-8 -*-
"""
Generate prompts based on the ontology
"""
import click
from rdflib import Graph
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_RDF_XML
from src.helpers.helpers import rdflib_to_pd

SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
PRED_TO_ABBR = {
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'rdf:type',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'rdfs:subClassOf'
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
    FILTER(?giv IN (cdo:<GIV>Variable))
} 
"""

PROMPT_TEMPLATE = """
In this task, you need to generate triples based on an ontology. There are constraints on how you can generate the hypotheses.

[ONTOLOGY]
```
<TRIPLES>
```

[TASK]
You need to generate the most sensible and coherent hypotheses in the form:
```
Cooperation is significantly <higher OR lower> when `SIV1` is `SIVV1`, compared to when `SIV2` is `SIVV2`.
```

[CONSTRAINTS]
1 - SIV1 and SIV2 must be SIV from the ontology, and SIVV1 and SIVV2 SIVV from the ontology. Entities of type  `SIV` are subjects of triples from the ontology with predicate `rdfs:subClassOf`.  Entities of type `SIVV` are subjects of triples from the ontology with predicate `rdf:type`.
2 - The hypothesis must be sensible and coherent.
3 - When generating the hypothesis, keep the bracket and the original names in the ontology.

Can you generate up to 5 sensible and coherent hypotheses?  You need to respect constraint 1 AND 2 AND 3.
"""

def write_triples(filtered):
    """ Add triples to prompt """
    res = []
    for _, row in filtered.iterrows():
        res.append(f"(cdo:{row.subject.split('/')[-1]}, {PRED_TO_ABBR[row.predicate]}, cdo:{row.object.split('/')[-1]})")
    return "\n".join(res)

@click.command()
@click.argument("variable")
@click.option("--save_path", help="path to save prompt")
def main(variable, save_path):
    """ Get full prompt """
    response = run_query(query=QUERY.replace("<GIV>", variable),
                         sparql_endpoint=SPARQL_ENDPOINT, headers=HEADERS_RDF_XML)
    graph = Graph()
    graph.parse(data=response.text, format="application/rdf+xml")
    df = rdflib_to_pd(graph=graph)
    print(df.predicate.unique())

    filtered = df[df.predicate.isin(PRED_TO_ABBR.keys())] \
        .sort_values(by="predicate", ascending=False)
    ontology = write_triples(filtered=filtered)
    prompt = PROMPT_TEMPLATE.replace("<TRIPLES>", ontology)
    print(prompt)

    if save_path:
        f = open(save_path, "w+", encoding="utf-8")
        f.write(prompt)
        f.close()



if __name__ == '__main__':
    main()
