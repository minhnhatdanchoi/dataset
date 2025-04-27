"""
This module contains patterns for entity extraction from unstructured text.
Each pattern is a regular expression designed to match specific entity types.
"""

# Dictionary of entity extraction patterns for different entity types
ENTITY_PATTERNS = {
    'Skill': [
        r'((?:skilled|proficient|expert|experienced|knowledge|know how|hands-on experience)\s+(?:in|with|of)\s+[\w\s,\.\-]+)',
        r'(responsible|dedicated|proactive|get along with everyone)',
        r'(database management|software test(?:er|ing)|web (?:and|&) mobile (?:application )?development)',
        r'(implementing\s+[\w\s,\.\-]+)',
        r'(configuring\s+[\w\s,\.\-]+)',
        r'(deploying\s+[\w\s,\.\-]+)',
        r'(managing\s+[\w\s,\.\-]+)',
        r'(creating\s+[\w\s,\.\-]+)'
    ],
    'Technology': [
        # Programming Languages
        r'\b(Python|Java|C\+\+|JavaScript|TypeScript|SQL|Ruby|PHP|Go|Swift|Kotlin|Scala|R|MATLAB|Perl|Shell|Bash|PowerShell|HTML|CSS|C#|Objective-C|Assembly|Rust|Dart|Lua|Haskell|Elixir|Clojure|Groovy|F#|C)\b',
        # Tools
        r'\b(Git|SVN|Jira|Redmine|Trello|Asana|Confluence|Slack|Teams|Grafana|Jenkins|CircleCI|Travis|SonarQube|Postman|Swagger|Kibana|Logstash|IntelliJ|VSCode|Eclipse|Xcode|Figma|Sketch|Photoshop|Illustrator)\b',
        r'\b(Redmine|Grafana|Jira)(?:[,\s]+(?:and\s+)?(Redmine|Grafana|Jira))*\b',
        # Automation Tools
        r'\b(Ansible|Puppet|Chef|Terraform|CloudFormation|Jenkins|GitHub Actions|GitLab CI|Bamboo|ArgoCD|Spinnaker|Azure DevOps|Kubernetes|Docker|Selenium|Cypress|Pytest|JUnit|TestNG|Mocha|Jest|CI\/CD|automation test)\b',
        # Microservices
        r'\b(Microservices|API Gateway|Service Mesh|Istio|Envoy|gRPC|REST|GraphQL|Kafka|RabbitMQ|ActiveMQ|NATS|ZeroMQ|etcd|Consul|Eureka|Ribbon|Hystrix|Resilience4j)\b',
        # OS
        r'\b(Linux|Ubuntu|Debian|CentOS|RHEL|Fedora|Windows|Windows Server|macOS|iOS|Android|Unix|FreeBSD|OpenBSD|Chrome OS|Operation System|Operating System)\b',
        # Databases
        r'\b(MySQL|PostgreSQL|Oracle|SQL Server|MongoDB|Cassandra|Redis|Elasticsearch|Neo4j|DynamoDB|Cosmos DB|Firebase|Firestore|CouchDB|MariaDB|SQLite|InfluxDB|TimescaleDB|Snowflake|BigQuery|Redshift)\b',
        # Cloud
        r'\b(AWS|Amazon Web Services|EC2|S3|Lambda|ECS|EKS|RDS|DynamoDB|CloudFront|Route53|IAM|Azure|Microsoft Azure|VM|Blob Storage|Functions|AKS|SQL Database|Cosmos DB|GCP|Google Cloud|Compute Engine|Cloud Storage|Cloud Functions|GKE|BigQuery|Cloud SQL|Heroku|Digital Ocean|Alibaba Cloud|IBM Cloud|cloud(?:-based)?(?:\s+platforms)?|cloud platforms)\b',
        # Frameworks
        r'\b(Spring|Spring Boot|Django|Flask|Rails|Angular|React|Vue|Laravel|Symphony|Express|ASP\.NET|jQuery|Bootstrap|TensorFlow|PyTorch|Keras|scikit-learn|Hadoop|Spark|Apache Spark)\b',
        r'(Spring FrameWork)',
        # Platforms
        r'(Bidata platform|big data technologies)'
    ],
    'Experience_Level': [
        r'(\d+\+?\s*years?\s+(?:of\s+)?experience\s+(?:in|with|of)\s+[\w\s,\.\-]+)'
    ],
    'Project': [
        r'(worked\s+on\s+[\w\s,\.]+\s+project)',
        r'(developed\s+[\w\s,\.]+)',
        r'(implemented\s+[\w\s,\.]+)',
        r'(project\s+(?:called|named)\s+[\w\s,\.]+)'
    ],
    'Soft_Skill': [
        r'(actively learn|improve from colleagues|willing to contribute|get along with everyone)',
        r'(responsible|dedicated|proactive)'
    ]
}