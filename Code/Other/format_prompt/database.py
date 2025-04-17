examples_database = [
    {
        "question": "Find employees who know PostgresSQL",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'PostgresSQL'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with MySQL?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'MySQL'}}) RETURN e.name"
    },
    {
        "question": "Who knows MongoDB?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'MongoDB'}}) RETURN e.name"
    },
    {
        "question": "Get employees that are proficient in DMS",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'DMS'}}) RETURN e.name"
    },
    {
        "question": "List employees who have experience with Oracle",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Oracle'}}) RETURN e.name"
    },
    {
        "question": "Which employees are skilled in Athena?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Athena'}}) RETURN e.name"
    },
    {
        "question": "Find all employees who know Redshift",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Redshift'}}) RETURN e.name"
    },
    {
        "question": "Which employees know Oracle SQL?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Oracle SQL'}}) RETURN e.name"
    },
    {
        "question": "Who is familiar with DynamoDB?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'DynamoDB'}}) RETURN e.name"
    },
    {
        "question": "Find employees who know PosgreSQL",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'PosgreSQL'}}) RETURN e.name"
    },
    {
        "question": "Which employees are experienced in SQL Server?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'SQL Server'}}) RETURN e.name"
    },
    {
        "question": "Find all employees who are skilled in PostgreSQL",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'PostgreSQL'}}) RETURN e.name"
    },
    {
        "question": "Who knows Cassandra?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Cassandra'}}) RETURN e.name"
    },
    {
        "question": "List employees who have worked with DocumentDB",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'DocumentDB'}}) RETURN e.name"
    },
    {
        "question": "Who is proficient in RDS?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'RDS'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with Redis?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Redis'}}) RETURN e.name"
    },
    {
        "question": "Find employees who know MariaDB",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'MariaDB'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with both PostgresSQL and MySQL?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'PostgresSQL'}}), (e)-[:DATABASE]->(db2:Database {{name: 'MySQL'}}) RETURN e.name"
    },
    {
        "question": "Find employees who know both MongoDB and DMS",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'MongoDB'}}), (e)-[:DATABASE]->(db2:Database {{name: 'DMS'}}) RETURN e.name"
    },
    {
        "question": "List employees who are proficient in Oracle and Athena",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Oracle'}}), (e)-[:DATABASE]->(db2:Database {{name: 'Athena'}}) RETURN e.name"
    },
    {
        "question": "Which employees have worked with Redshift and Oracle SQL?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'Redshift'}}), (e)-[:DATABASE]->(db2:Database {{name: 'Oracle SQL'}}) RETURN e.name"
    },
    {
        "question": "Find employees familiar with both DynamoDB and PosgreSQL",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'DynamoDB'}}), (e)-[:DATABASE]->(db2:Database {{name: 'PosgreSQL'}}) RETURN e.name"
    },
    {
        "question": "Which employees know SQL Server and PostgreSQL?",
        "query": "MATCH (e:Employee)-[:DATABASE]->(db:Database {{name: 'SQL Server'}}), (e)-[:DATABASE]->(db2:Database {{name: 'PostgreSQL'}}) RETURN e.name"
    }
]
