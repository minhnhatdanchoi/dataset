examples_seniority = [
    {
        "question": "Find employees with more than 3 years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) > 3 RETURN e.name"
    },
    {
        "question": "Which employees have more than 5 years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) > 5 RETURN e.name"
    },
    {
        "question": "Who has more than 2 years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) > 2 RETURN e.name"
    },
    {
        "question": "Find employees with more than 6 years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) > 6 RETURN e.name"
    },
    {
        "question": "Which employees have more than 9 years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) > 9 RETURN e.name"
    },
    {
        "question": "Find employees with less than 5 years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) < 5 RETURN e.name"
    },
    {
        "question": "Who has less than 3 years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) < 3 RETURN e.name"
    },
    {
        "question": "Find employees with less than 7 years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) < 7 RETURN e.name"
    },
    {
        "question": "Which employees have 3 years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) = 3 RETURN e.name"
    },
    {
        "question": "Find employees with 6 or more years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) >= 6 RETURN e.name"
    },
    {
        "question": "Which employees have 2 or more years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) >= 2 RETURN e.name"
    },
    {
        "question": "Who has 4 or more years of experience?",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) >= 4 RETURN e.name"
    },
    {
        "question": "Find employees with exactly 9 years of experience",
        "query": "MATCH (e:Employee)-[:HAS_SENIORITY]->(n:Seniority) WHERE toInteger(replace(n.name, ' year', '')) = 9 RETURN e.name"
    }
]