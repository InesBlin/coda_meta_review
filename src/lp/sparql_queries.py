# -*- coding: utf-8 -*-
"""
For more clarity, storing all SPARQL queries here
"""
from kglab.helpers.variables import STR_RDF, STR_RDFS

PREFIXES = f"""
PREFIX cp: <https://data.cooperationdatabank.org/vocab/prop/>
PREFIX rdf: <{STR_RDF}>
PREFIX rdfs: <{STR_RDFS}>
PREFIX id: <https://data.cooperationdatabank.org/id/>
PREFIX class: <https://data.cooperationdatabank.org/vocab/class/>
"""

TREATMENT_VALS_T = """
SELECT ?t1 ?t2 ?p ?o1 ?o2 WHERE {
    ?t1 ?p ?o1 .
    ?t2 ?p ?o2 .
    VALUES ?t1 {<[iri1]>}
    VALUES ?t2 {<[iri2]>}
}
"""

HB_REGULAR_T = PREFIXES + """
SELECT * WHERE {
  
  ?obs rdf:type class:Observation .
  ?obs cp:eSmeasure <https://data.cooperationdatabank.org/id/esmeasure/[es_measure]> . 
  ?obs cp:dependentVariable ?dependent . 
  ?obs cp:eSEstimate ?ES .
  ?obs cp:effectSizeSampleSize ?N . 

  ?obs cp:treatment ?t1, ?t2 . 
  ?study cp:reportsEffect ?obs . 
  ?t1 cp:betweenOrWithinParticipantsDesign ?design . 


  OPTIONAL {
    ?t1 cp:nCondition ?n1 . 
    ?t2 cp:nCondition ?n2 .

    ?t1 cp:sDforCondition ?sd1 . 
    ?t2 cp:sDforCondition ?sd2 .
    } 

  OPTIONAL { 
    ?obs cp:effectSizeLowerLimit ?ESLower . 
    ?obs cp:effectSizeUpperLimit ?ESUpper .  
    }

  OPTIONAL {
    ?paper cp:study ?study . 
    ?paper cp:doi ?doi . }
    
  FILTER (STR(?t1) < STR(?t2)) 
} 
"""

EFFECT_CONSTRUCT_T = PREFIXES + """
CONSTRUCT {
  ?study cp:reportsEffect <[obs]> . 
  <[obs]> cp:dependentVariable ?dependentVariable .
  <[obs]> cp:ESType  class:[es_type] . 
  <[obs]> cp:treatment ?t1, ?t2 .
  <[iv]> rdfs:subPropertyOf ?superProperty . 
  
  [line_t1]
  [line_t2]
  [cp_effect]
  <[iv]> rdfs:range ?class . 
  ?class rdfs:subClassOf ?superClass . 
  ?superClass rdfs:subClassOf class:IndependentVariable . 
  <[iv_h]> rdfs:subPropertyOf  <[iv]> . 
} WHERE {
  ?study cp:reportsEffect <[obs]> . 
  <[obs]> rdf:type class:Observation .
  <[obs]> cp:eSmeasure <https://data.cooperationdatabank.org/id/esmeasure/[es_measure]> . 
  <[obs]> cp:dependentVariable ?dependentVariable .
  <[obs]> cp:treatment ?t1, ?t2 . 
<[iv]> rdfs:subPropertyOf ?superProperty . 
  
  OPTIONAL { 
    [line_t1]
    [line_t2]
  }
  OPTIONAL {
    <[iv]> rdfs:range ?class . 
    ?class rdfs:subClassOf ?superClass . 
    ?superClass rdfs:subClassOf class:IndependentVariable . 
  }
FILTER (STR(?t1) < STR(?t2)) 
}
"""