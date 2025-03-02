automation_tool_cypher=[
    {
        "question": "Find employees using GitlabCI",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool {{name: 'GitlabCI'}}) RETURN e.name"
    },
    {
        "question": "Find employees using Terraform",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool {{name: 'Terraform'}}) RETURN e.name"

    },
    {
        "question": "Find employees using both Jenkins and Terraform",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(t1:AutomationTool {{name: 'Jenkins'}}), (e)-[:AUTOTOOL]->(t2:AutomationTool {{name: 'Terraform'}}) RETURN e.name"
    },
    {
        "question": "Find employees who use any automation tool",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(t:AutomationTool) RETURN e.name, t.name"
    },
    {
        "question": "Find employees using Ansible",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool {{name: 'Ansible'}}) RETURN e.name"
    },
    {
        "question": "Find employees using AWS CloudFormation with expertise level 'High'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool {{name: 'AWS CloudFormation'}}) WHERE automationTool.expertise = 'High' RETURN e.name"
    },
    {
        "question": "Find employees using ArgoCD or GitlabCI",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool) WHERE automationTool.name IN ['ArgoCD', 'GitlabCI'] RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using automation tools related to AWS",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool) WHERE automationTool.name STARTS WITH 'AWS' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find all automation tools used by employees",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool) RETURN DISTINCT automationTool.name"
    },
    {
        "question": "Find employees using any tool that contains 'AWS'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'AWS' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using any tool that contains 'CI'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'CI' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using tools that contain 'CD' (e.g., ArgoCD, Gitlab CI/CD)",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'CD' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using any automation tool with 'Form' in the name",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'Form' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using automation tools that contain 'Test'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'Test' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using automation tools that contain 'Ansible'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'Ansible' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using any tool with 'Pipeline' in its name",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'Pipeline' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using automation tools that contain 'Deploy'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.name CONTAINS 'Deploy' RETURN e.name, automationTool.name"
    },
    {
        "question": "Find employees using automation tools with expertise level containing 'High'",
        "query": "MATCH (e:Employee)-[:AUTOTOOL]->(automationTool:AutomationTool) WHERE automationTool.expertise CONTAINS 'High' RETURN e.name, automationTool.name"
    }
]

