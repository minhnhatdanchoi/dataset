examples_program_language = [
    {
        "question": "Find employees who know Python",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Python'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with Java?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java'}}) RETURN e.name"
    },
    {
        "question": "Who knows SQL?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'SQL'}}) RETURN e.name"
    },
    {
        "question": "Get employees that are proficient in C++",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'C++'}}) RETURN e.name"
    },
    {
        "question": "List employees who have experience with HTML",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'HTML'}}) RETURN e.name"
    },
    {
        "question": "Which employees are skilled in Java Spring Boot?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java Spring Boot'}}) RETURN e.name"
    },
    {
        "question": "Find all employees who know Java Core",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java Core'}}) RETURN e.name"
    },
    {
        "question": "Which employees know both Python and SQL?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Python'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'SQL'}}) RETURN e.name"
    },
    {
        "question": "Who is familiar with CSS?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'CSS'}}) RETURN e.name"
    },
    {
        "question": "Find employees who have experience with TypeScript",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'TypeScript'}}) RETURN e.name"
    },
    {
        "question": "Which employees are experienced in Node.js?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Nodejs'}}) RETURN e.name"
    },
    {
        "question": "List employees who are proficient in C#",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'C#'}}) RETURN e.name"
    },
    {
        "question": "Who knows Kotlin?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Kotlin'}}) RETURN e.name"
    },
    {
        "question": "Find all employees who are skilled in JavaScript",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Javascript'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with PowerShell?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'PowerShell'}}) RETURN e.name"
    },
    {
        "question": "Find employees who know TypeScript and JavaScript",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'TypeScript'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'Javascript'}}) RETURN e.name"
    },
    {
        "question": "Who has experience in Angular?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Angular'}}) RETURN e.name"
    },
    {
        "question": "List employees who have worked with Vue.js",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'VueJs'}}) RETURN e.name"
    },
    {
        "question": "Who is proficient in PHP?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'PHP'}}) RETURN e.name"
    },
    {
        "question": "Which employees know Java and Kotlin?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'Kotlin'}}) RETURN e.name"
    },
    {
        "question": "Find employees who know .NET",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: '.NET'}}) RETURN e.name"
    },
    {
        "question": "Which employees have worked with Hive SQL?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'HiveSql'}}) RETURN e.name"
    },
    {
        "question": "Get employees familiar with Scala",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Scala'}}) RETURN e.name"
    },
    {
        "question": "List employees who are familiar with Bash scripting",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Bash Script'}}) RETURN e.name"
    },
    {
        "question": "Find employees skilled in C programming language",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'C'}}) RETURN e.name"
    },
    {
        "question": "Which employees are skilled in Java and Python?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'Python'}}) RETURN e.name"
    },
    {
        "question": "Who is proficient in C++ and JavaScript?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'C++'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'Javascript'}}) RETURN e.name"
    },
    {
        "question": "Find employees who have worked with both HTML and CSS",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'HTML'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'CSS'}}) RETURN e.name"
    },
    {
        "question": "Which employees are familiar with Java Spring Boot and SQL?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java Spring Boot'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'SQL'}}) RETURN e.name"
    },
    {
        "question": "List employees who have experience in Java and C#",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Java'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'C#'}}) RETURN e.name"
    },
    {
        "question": "Who knows Python and C#?",
        "query": "MATCH (e:Employee)-[:PROGRAM_LANGUAGE]->(pl:ProgramingLanguage {{name: 'Python'}}), (e)-[:PROGRAM_LANGUAGE]->(pl2:ProgramingLanguage {{name: 'C#'}}) RETURN e.name"
    }
]
