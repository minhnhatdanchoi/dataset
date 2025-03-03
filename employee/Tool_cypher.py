examples_tool = [
    {
        "question": "Find employees who use ELK Stack",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool {{name: 'ELK Stack'}}) RETURN e.name"
    },
    {
        "question": "Which employees are working with Prometheus?",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool {{name: 'Prometheus'}}) RETURN e.name"
    },
    {
        "question": "List employees who use IntelliJ",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool {{name: 'IntelliJ'}}) RETURN e.name"
    },
    {
        "question": "Find employees who use both Grafana and AWS CloudWatch",
        "query": "MATCH (e:Employee)-[:TOOL]->(t1:Tool {{name: 'Grafana'}}), (e)-[:TOOL]->(t2:Tool {{name: 'AWS CloudWatch'}}) RETURN e.name"
    },
    {
        "question": "Show employees using either VSCode or Eclipse",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool) WHERE t.name IN ['VSCode', 'Eclipse'] RETURN e.name"
    },
    {
        "question": "Find employees who are proficient in all tools",
        "query": "MATCH (e:Employee) WHERE ALL(t IN ['ELK Stack', 'Prometheus', 'NodeExporter', 'Grafana', 'AWS Cloudwatch', 'AWS Lambda', 'Databrick', 'Hadoop', 'Kafka', 'AWS Glue', 'Superset', 'Apache Spark', 'Graylog', 'IntelliJ', 'Visual Studio Code', 'Git', 'draw.io', 'Eclipse', 'Redmine', 'SVN', 'Backlog', 'MS Office', 'Postman', 'Jira', 'Dynatrace', 'Kibana Elastic Search', 'Jmeter', 'backlog', 'Netbeans', 'Visual Studio', 'VSCode', 'PHPStorm', 'Notepad++', 'Figma', 'Adobe XD', 'Gitlab', 'Vangrind', 'ERP', 'PHP storm', 'Trello', 'BitBucket', 'Odoo', 'Sketch', 'SonarQube', 'Zeplin', 'Android Studio', 'CloudWatch', 'AWS CloudWatch', 'New Relic', 'NetApp', 'Pub-Sub', 'Cloud Monitoring'] WHERE (e)-[:TOOL]->(:Tool {{name: t}})) RETURN e.name"
    },
    {
        "question": "Which employees are using at least one tool?",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool) RETURN DISTINCT e.name"
    },
    {
        "question": "Find employees who use IntelliJ but not Visual Studio Code",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool {{name: 'IntelliJ'}}) WHERE NOT (e)-[:TOOL]->(:Tool {{name: 'Visual Studio Code'}}) RETURN e.name"
    },
    {
        "question": "Which employees use both Git and Jira but not SVN?",
        "query": "MATCH (e:Employee)-[:TOOL]->(t1:Tool {{name: 'Git'}}), (e)-[:TOOL]->(t2:Tool {{name: 'Jira'}}) WHERE NOT (e)-[:TOOL]->(:Tool {{name: 'SVN'}}) RETURN e.name"
    },
    {
        "question": "List all employees and their tools",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool) RETURN e.name, t.name"
    },
    {
        "question": "Count how many employees are using Postman",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool {{name: 'Postman'}}) RETURN COUNT(e) AS postman_users"
    },
    {
        "question": "Find employees who are using only one tool",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool) WITH e, COUNT(t) AS tool_count WHERE tool_count = 1 RETURN e.name"
    },
    {
        "question": "Which employees use multiple tools?",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool) WITH e, COUNT(t) AS tool_count WHERE tool_count > 1 RETURN e.name"
    },
    {
        "question": "Find employees who use Jmeter and another tool",
        "query": "MATCH (e:Employee)-[:TOOL]->(t1:Tool {{name: 'Jmeter'}}), (e)-[:TOOL]->(t2:Tool) WHERE t2.name <> 'Jmeter' RETURN e.name"
    },
    {
        "question": "Find tools and their corresponding levels",
        "query": "MATCH (t:Tool)-[:HAS_LEVEL]->(l:Level) RETURN t.name, l.name"
    },
    {
        "question": "Find employees using a tool with a specific level",
        "query": "MATCH (e:Employee)-[:TOOL]->(t:Tool)-[:HAS_LEVEL]->(l:Level {{name: 'Advanced'}}) RETURN e.name, t.name"
    }
]