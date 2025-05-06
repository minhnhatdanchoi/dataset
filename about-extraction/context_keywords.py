"""
This module contains context keywords used for evaluating the relevance
of extracted entities within their textual context.
"""

# Dictionary of context keywords for different entity types
CONTEXT_KEYWORDS = {
    'Skill': [
        "experience", "expert", "skill", "proficient", "knowledge",
        "know", "skilled", "responsible", "dedicated"
    ],
    'Technology': [
        "Python", "Java", "C++", "JavaScript", "SQL", "code", "develop", "program", "C", "tool",
        "use", "using", "Jira", "Redmine", "Grafana", "manage", "automation", "CI/CD", "pipeline",
        "deploy", "terraform", "kubernetes", "ArgoCD", "microservice", "service", "API", "REST",
        "gRPC", "OS", "system", "platform", "Linux", "Windows", "Operation System", "database", "DB",
        "data", "SQL", "NoSQL", "query", "cloud", "AWS", "Azure", "GCP", "service", "infrastructure",
        "resources", "spring", "framework", "hadoop", "spark", "apache", "platform", "bidata",
        "big data"
    ],
    'Experience_Level': [
        "years", "year", "experience", "worked", "+"
    ],
    'Project': [
        "project", "worked on", "involved in", "developed"
    ],
    'Soft_Skill': [
        "learn", "improve", "contribute", "proactive",
        "responsible", "dedicated", "get along"
    ]
}