examples_microservice = [
    {
        "question": "Find employees who use Docker",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice {{name: 'Docker'}}) RETURN e.name"
    },
    {
        "question": "Which employees are working with Kubernetes?",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice {{name: 'Kubernetes'}}) RETURN e.name"
    },
    {
        "question": "List employees who work with ECS",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice {{name: 'ECS'}}) RETURN e.name"
    },
    {
        "question": "Find employees who use both EKS and S3",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m1:Microservice {{name: 'EKS'}}), (e)-[:MICROSERVICE]->(m2:Microservice {{name: 'S3'}}) RETURN e.name"
    },
    {
        "question": "Show employees using either EC2 or ALB",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice) WHERE m.name IN ['EC2', 'ALB'] RETURN e.name"
    },
    {
        "question": "Find employees who are proficient in all microservices",
        "query": "MATCH (e:Employee) WHERE ALL(m IN ['Docker', 'Kubernetes', 'ECS', 'ECR', 'Openshift', 'Podman', 'Kafka', 'EKS', 'K8s', 'S3', 'NLB', 'AWS Services: EKS', 'Rancher', 'CloudFront', 'Microservice', 'Helm Chart', 'EC2', 'ALB', 'API Gateway'] WHERE (e)-[:MICROSERVICE]->(:Microservice {{name: m}})) RETURN e.name"
    },
    {
        "question": "Which employees are using at least one microservice?",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice) RETURN DISTINCT e.name"
    },
    {
        "question": "Find employees who use OpenShift but not EKS",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice {{name: 'Openshift'}}) WHERE NOT (e)-[:MICROSERVICE]->(:Microservice {{name: 'EKS'}}) RETURN e.name"
    },
    {
        "question": "Which employees use both API Gateway and Kafka but not EC2?",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m1:Microservice {{name: 'API Gateway'}}), (e)-[:MICROSERVICE]->(m2:Microservice {{name: 'Kafka'}}) WHERE NOT (e)-[:MICROSERVICE]->(:Microservice {{name: 'EC2'}}) RETURN e.name"
    },
    {
        "question": "List all employees and their microservices",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice) RETURN e.name, m.name"
    },
    {
        "question": "Count how many employees are using Kubernetes",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice {{name: 'Kubernetes'}}) RETURN COUNT(e) AS kubernetes_users"
    },
    {
        "question": "Find employees who are using only one microservice",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice) WITH e, COUNT(m) AS microservice_count WHERE microservice_count = 1 RETURN e.name"
    },
    {
        "question": "Which employees use multiple microservices?",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m:Microservice) WITH e, COUNT(m) AS microservice_count WHERE microservice_count > 1 RETURN e.name"
    },
    {
        "question": "Find employees who use ALB and another microservice",
        "query": "MATCH (e:Employee)-[:MICROSERVICE]->(m1:Microservice {{name: 'ALB'}}), (e)-[:MICROSERVICE]->(m2:Microservice) WHERE m2.name <> 'ALB' RETURN e.name"
    }
]