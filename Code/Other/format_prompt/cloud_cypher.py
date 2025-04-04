examples_cloud = [
    {
        "question": "Find employees who use AWS",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud {{name: 'AWS'}}) RETURN e.name"
    },
    {
        "question": "Which employees are working with GCP?",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud {{name: 'GCP'}}) RETURN e.name"
    },
    {
        "question": "List employees who work with Azure",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud {{name: 'Azure'}}) RETURN e.name"
    },
    {
        "question": "Find employees who use both AWS and GCP",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c1:Cloud {{name: 'AWS'}}), (e)-[:CLOUD]->(c2:Cloud {{name: 'GCP'}}) RETURN e.name"
    },
    {
        "question": "Show employees using either AWS or Azure",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud) WHERE c.name IN ['AWS', 'Azure'] RETURN e.name"
    },
    {
        "question": "Find employees who are certified in all three cloud providers",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c1:Cloud {{name: 'AWS'}}), (e)-[:CLOUD]->(c2:Cloud {{name: 'GCP'}}), (e)-[:CLOUD]->(c3:Cloud {{name: 'Azure'}}) RETURN e.name"
    },
    {
        "question": "Which employees are using at least one cloud provider?",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud) RETURN DISTINCT e.name"
    },
    {
        "question": "Find employees who work with GCP but not AWS",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud {{name: 'GCP'}}) WHERE NOT (e)-[:CLOUD]->(:Cloud {{name: 'AWS'}}) RETURN e.name"
    },
    {
        "question": "Which employees use both Azure and AWS but not GCP?",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c1:Cloud {{name: 'Azure'}}), (e)-[:CLOUD]->(c2:Cloud {{name: 'AWS'}}) WHERE NOT (e)-[:CLOUD]->(:Cloud {{name: 'GCP'}}) RETURN e.name"
    },
    {
        "question": "List all employees and their cloud providers",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud) RETURN e.name, c.name"
    },
    {
        "question": "Count how many employees are using AWS",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud {{name: 'AWS'}}) RETURN COUNT(e) AS aws_users"
    },
    {
        "question": "Find employees who are only using one cloud provider",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud) WITH e, COUNT(c) AS cloud_count WHERE cloud_count = 1 RETURN e.name"
    },
    {
        "question": "Which employees use multi-cloud (more than one provider)?",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c:Cloud) WITH e, COUNT(c) AS cloud_count WHERE cloud_count > 1 RETURN e.name"
    },
    {
        "question": "Find employees who use AWS and another cloud provider",
        "query": "MATCH (e:Employee)-[:CLOUD]->(c1:Cloud {{name: 'AWS'}}), (e)-[:CLOUD]->(c2:Cloud) WHERE c2.name <> 'AWS' RETURN e.name"
    }
]