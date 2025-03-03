examples_os = [
    {
        "question": "Find employees who use Windows",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'Windows'}}) RETURN e.name"
    },
    {
        "question": "Which employees are working with Ubuntu?",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'Ubuntu'}}) RETURN e.name"
    },
    {
        "question": "List employees who use Linux",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'Linux'}}) RETURN e.name"
    },
    {
        "question": "Find employees who work with MacOS",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'MacOS'}}) RETURN e.name"
    },
    {
        "question": "Show employees using both Windows and Ubuntu",
        "query": "MATCH (e:Employee)-[:OS]->(o1:OS {{name: 'Windows'}}), (e)-[:OS]->(o2:OS {{name: 'Ubuntu'}}) RETURN e.name"
    },
    {
        "question": "Find employees using either Linux or MacOS",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS) WHERE o.name IN ['Linux', 'MacOS'] RETURN e.name"
    },
    {
        "question": "Find employees who are proficient in all four operating systems",
        "query": "MATCH (e:Employee)-[:OS]->(o1:OS {{name: 'Windows'}}), (e)-[:OS]->(o2:OS {{name: 'Ubuntu'}}), (e)-[:OS]->(o3:OS {{name: 'Linux'}}), (e)-[:OS]->(o4:OS {{name: 'MacOS'}}) RETURN e.name"
    },
    {
        "question": "Which employees are using at least one operating system?",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS) RETURN DISTINCT e.name"
    },
    {
        "question": "Find employees who use Ubuntu but not Windows",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'Ubuntu'}}) WHERE NOT (e)-[:OS]->(:OS {{name: 'Windows'}}) RETURN e.name"
    },
    {
        "question": "Which employees use both MacOS and Linux but not Windows?",
        "query": "MATCH (e:Employee)-[:OS]->(o1:OS {{name: 'MacOS'}}), (e)-[:OS]->(o2:OS {{name: 'Linux'}}) WHERE NOT (e)-[:OS]->(:OS {{name: 'Windows'}}) RETURN e.name"
    },
    {
        "question": "List all employees and their operating systems",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS) RETURN e.name, o.name"
    },
    {
        "question": "Count how many employees are using Windows",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS {{name: 'Windows'}}) RETURN COUNT(e) AS windows_users"
    },
    {
        "question": "Find employees who are using only one operating system",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS) WITH e, COUNT(o) AS os_count WHERE os_count = 1 RETURN e.name"
    },
    {
        "question": "Which employees use multiple operating systems?",
        "query": "MATCH (e:Employee)-[:OS]->(o:OS) WITH e, COUNT(o) AS os_count WHERE os_count > 1 RETURN e.name"
    },
    {
        "question": "Find employees who use Windows and another operating system",
        "query": "MATCH (e:Employee)-[:OS]->(o1:OS {{name: 'Windows'}}), (e)-[:OS]->(o2:OS) WHERE o2.name <> 'Windows' RETURN e.name"
    }
]