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

TREATMENT_VALS_T_REGULAR = PREFIXES + """
SELECT * WHERE {
    ?t1 ?iv ?cat_t1 .
    ?t2 ?iv ?cat_t2 .
    ?iv rdfs:subPropertyOf ?giv_prop .
    OPTIONAL {
      ?iv rdfs:range ?range_class_iv .
      ?range_class_iv rdfs:subClassOf ?range_superclass_iv .
      ?range_superclass_iv rdfs:subClassOf class:IndependentVariable .
    }
    #OPTIONAL {?iv rdfs:label ?iv_label .}
    #OPTIONAL {?cat_t1 rdfs:label ?cat_t1_label .}
    #OPTIONAL {?cat_t2 rdfs:label ?cat_t2_label .}
    VALUES ?t1 {<[iri1]>}
    VALUES ?t2 {<[iri2]>}
    FILTER(?iv != ?giv_prop)
    FILTER(?cat_t1 != ?cat_t2)
}
"""

TREATMENT_VALS_T_VAR_MOD = PREFIXES + """
SELECT * WHERE {
    ?t1 ?iv ?cat_t1 ;
        ?mod ?mod_t1 .
    ?t2 ?iv ?cat_t2 ;
        ?mod ?mod_t2 .
    ?iv rdfs:subPropertyOf ?giv_prop .
    ?mod rdfs:subPropertyOf ?giv_prop .  
    OPTIONAL {
      ?iv rdfs:range ?range_class_iv .
      ?range_class_iv rdfs:subClassOf ?range_superclass_iv .
      ?range_superclass_iv rdfs:subClassOf class:IndependentVariable .
    }
    #OPTIONAL {?iv rdfs:label ?iv_label .}
    #OPTIONAL {?cat_t1 rdfs:label ?cat_t1_label .}
    #OPTIONAL {?cat_t2 rdfs:label ?cat_t2_label .}
    #OPTIONAL {?mod rdfs:label ?mod_label .}
    #OPTIONAL {?mod_t1 rdfs:label ?mod_t1_label .}
    #OPTIONAL {?mod_t2 rdfs:label ?mod_t2_label .}
    OPTIONAL {
      ?mod rdfs:range ?range_class_mod .
      ?range_class_mod rdfs:subClassOf ?range_superclass_mod .
      ?range_superclass_mod rdfs:subClassOf class:IndependentVariable .  
    }
    VALUES ?t1 {<[iri1]>}
    VALUES ?t2 {<[iri2]>}
    FILTER(?iv != ?mod)
    FILTER(?iv != ?giv_prop)
    FILTER(?mod != ?giv_prop)
    FILTER(?cat_t1 != ?cat_t2)
    FILTER(?mod_t1 != ?mod_t2)
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

HB_STUDY_T = PREFIXES + """
SELECT * WHERE {
  
  ?obs rdf:type class:Observation .
  ?obs cp:eSmeasure <https://data.cooperationdatabank.org/id/esmeasure/[es_measure]> . 
  ?obs cp:dependentVariable ?dependent . 
  ?obs cp:eSEstimate ?ES .
  ?obs cp:effectSizeSampleSize ?N . 

  ?obs cp:treatment ?t1, ?t2 . 
  ?study cp:reportsEffect ?obs ;
         ?mod ?mod_val .
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
  
  #OPTIONAL {?mod rdfs:label ?mod_label .}
  #OPTIONAL {?mod_val rdfs:label ?mod_val_label .}
    
  FILTER (STR(?t1) < STR(?t2)) 
  FILTER (?mod NOT IN (rdf:type, rdfs:label, cp:comments, cp:descriptionIV,
        			   cp:otherVariables, cp:reportsEffect,  cp:studyOtherDilemmaType,
        			   cp:studySequentiality))

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

ONTOLOGY_QUERY = PREFIXES + """
SELECT ?s ?p ?o
#FROM NAMED <https://data.cooperationdatabank.org/countryVocab>
FROM NAMED <https://data.cooperationdatabank.org/Vocab>
WHERE {
    GRAPH ?graph {
        ?s ?p ?o .
    }
}
"""

LABELS_QUERY = PREFIXES + """
SELECT ?n ?nl WHERE {
  ?n rdfs:label ?nl .
}
"""