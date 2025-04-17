examples_language= [
    {
        "question": "Find employees who speak Vietnamese",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'Vietnamese'}}) Return e.name"
    },
    {
        "question": "Person speak Vietnamese",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'Vietnamese'}}) Return e.name"
    },
    {
        "question": "Which workers are proficient in French?",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'French'}}) Return e.name"
    },
    {
        "question": "Show employees who speak both English and German",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'English'}}), (e)-[:LANGUAGE]->(l2:Language {{name: 'German'}}) Return e.name"
    },
    {
        "question": "List all employees who speak Japanese or Korean",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language) WHERE l.name IN ['Japanese', 'Korean'] Return e.name"
    },
    {
        "question": "Which employees have Spanish as their second language?",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'Spanish'}}) WHERE e.language_level = 'Second' Return e.name"
    },
    {
        "question": "Find employees who are fluent in Mandarin",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'Mandarin'}}) AND e.language_level = 'Fluent' Return e.name"
    },
    {
        "question": "Which employees have German as their native language?",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'German'}}) WHERE e.language_level = 'Native' Return e.name"
    },
    {
        "question": "Find employees who speak English and have a TOEFL score of 100+",
        "query": "MATCH (e:Employee)-[:LANGUAGE]->(l:Language {{name: 'English'}}), (e)-[:HAS_LANGUAGE_LEVEL]->(le:Level {{name: 'TOEFL 100+'}}) Return e.name"
    },
    {
        "question": "Which workers have a TOEIC score of 700?",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level {{name: 'Toeic 700'}}) Return e.name"
    },
    {
        "question": "Find employees with TOEIC 800 or higher",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level) WHERE le.name IN ['Toeic 800', 'Toeic 850', 'Toeic 900'] Return e.name"
    },
    {
        "question": "Which employees have a TOEFL score of 90?",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level {{name: 'TOEFL 90'}}) Return e.name"
    },
    {
        "question": "Find workers who have IELTS scores between 6.0 and 7.0",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level) WHERE le.name IN ['IELTS 6.0', 'IELTS 6.5', 'IELTS 7.0'] Return e.name"
    },
    {
        "question": "Which employees have a language level higher than B2 in English?",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level) WHERE le.name IN ['C1', 'C2'] AND EXISTS {{MATCH (e)-[:LANGUAGE]->(l:Language {{name: 'English'}})}} Return e.name"
    },
    {
        "question": "Find employees who have a language level of B2 or higher in French",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level) WHERE le.name IN ['B2', 'C1', 'C2'] AND EXISTS {{MATCH (e)-[:LANGUAGE]->(l:Language {{name: 'French'}})}} Return e.name"
    },
    {
        "question": "List employees who have a CEFR level C1 in English",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level {{name: 'C1'}})-[:LEVEL_OF_LANGUAGE]->(l:Language {{name: 'English'}}) Return e.name"
    },
    {
        "question": "Which employees have a language proficiency level of A1 in any language?",
        "query": "MATCH (e:Employee)-[:HAS_LANGUAGE_LEVEL]->(le:Level {{name: 'A1'}}) Return e.name"
    }
]
