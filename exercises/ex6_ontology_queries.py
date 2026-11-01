"""Exercise 6 (Topic 2 / conceptual representation): querying the ontology with SPARQL."""
from pathlib import Path

from rdflib import Graph

g = Graph().parse(Path(__file__).resolve().parent.parent / "ontology" / "softtek_ontology.ttl")
print(f"Ontology loaded: {len(g)} triples\n")

QUESTIONS = {
    "Q1. Which document types exist?": """
        SELECT ?label WHERE { ?t rdfs:subClassOf ex:Document ; rdfs:label ?label } ORDER BY ?label""",
    "Q2. Which documents relate to the Payments API (directly or through an incident)?": """
        SELECT DISTINCT ?id ?title ?how WHERE {
          { ?d ex:describes ex:PaymentsAPI . BIND("describes the system" AS ?how) }
          UNION { ?d ex:documents ?i . ?i ex:affects ex:PaymentsAPI . BIND("documents an incident on it" AS ?how) }
          ?d ex:docId ?id ; rdfs:label ?title } ORDER BY ?id""",
    "Q3. Which documents must be updated because of lessons learned from incidents?": """
        SELECT ?incident ?lesson ?doc WHERE {
          ?i ex:resultsIn ?l ; rdfs:label ?incident . ?l rdfs:label ?lesson ; ex:updates ?d .
          ?d rdfs:label ?doc }""",
    "Q4. What should a new SRE read to operate Kubernetes? (skill -> training -> referenced docs)": """
        SELECT ?training ?ref WHERE {
          ?s ex:skillIn ex:Kubernetes . ?t ex:teaches ?s ; rdfs:label ?training .
          OPTIONAL { ?t ex:references ?r . ?r rdfs:label ?ref } }""",
    "Q5. How many documents does each department own?": """
        SELECT ?dept (COUNT(?d) AS ?n) WHERE { ?d ex:ownedBy ?x . ?x rdfs:label ?dept }
        GROUP BY ?dept ORDER BY DESC(?n)""",
}

for question, sparql in QUESTIONS.items():
    print(question)
    for row in g.query(sparql, initNs={"ex": "http://softtek.example/kms#",
                                       "rdfs": "http://www.w3.org/2000/01/rdf-schema#"}):
        print("   ", " | ".join(str(v) for v in row))
    print()
