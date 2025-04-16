"""
so sánh với dùng ChatGPT
so sánh với bản trước
"""
from py2neo import Graph
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from neo4j import GraphDatabase
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

# 🔗 Connect to Neo4j
NEO4J_URI = "neo4j+s://fa2fd127.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "k6y0bLBbHmLw5g-lopuQFKvIsEvjyTig7Y2r-p7aPOc"

try:
  graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
  print("✅ Neo4j connection successful!")
except Exception as e:
  print(f"❌ Neo4j connection error: {e}")
  exit()

# 📌 Extract data from Neo4j
query = """
MATCH (e:Employee)-[r]->(n)
WHERE type(r) <> 'HAS_ABOUT' AND any(label IN labels(n) WHERE label IN ['Nationality', 'Language', 'Sex', 'Project', 'Technology', 'Seniority'])
RETURN id(e) AS source, id(n) AS target, e.name AS employee_name, labels(n)[0] AS node_type, n.name AS node_name, type(r) AS relationship_type
UNION
MATCH (e:Employee)-[:HAS_ABOUT]->(a:About)
RETURN id(e) AS source, id(a) AS target, e.name AS employee_name, 'About' AS node_type, a.type AS node_name, 'HAS_ABOUT' AS relationship_type
"""
result = graph.run(query).to_data_frame()

if result.empty:
  print("⚠ No Employee-About relationships found")
  exit()
else:
  print(f"Found {len(result)} relationships")

# 📌 Convert data to tensors
edge_index = torch.tensor(result[['source', 'target']].values,
                          dtype=torch.long).t().contiguous()
n_nodes = max(edge_index.flatten()).item() + 1
x = torch.randn(n_nodes, 64)  # Random feature vector for each node
data = Data(x=x, edge_index=edge_index)

# 📌 Create ID to entity mapping dictionaries
id_to_info = {}
for _, row in result.iterrows():
  id_to_info[row['source']] = {'name': row['employee_name'], 'type': 'Employee'}
  id_to_info[row['target']] = {'name': row['node_name'],
                               'type': row['node_type']}

# 📌 Generate negative samples
positive_edges = edge_index.t().tolist()
all_nodes = set(range(n_nodes))
negative_edges = []

while len(negative_edges) < len(positive_edges):
  src, tgt = random.sample(sorted(all_nodes), 2)
  if [src, tgt] not in positive_edges:
    negative_edges.append([src, tgt])

y = torch.cat(
    [torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
train_edge_index = torch.tensor(positive_edges + negative_edges,
                                dtype=torch.long).t().contiguous()
data.edge_index = edge_index  # Keep original edges for graph structure
train_y = y  # Y labels for training edges


# 📌 Define GraphSAGE model
class LinkPredictor(nn.Module):
  def __init__(self, in_channels, hidden_channels):
    super().__init__()
    self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=3)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_channels * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

  def forward(self, x, edge_index):
    x = self.sage(x, edge_index)
    return x

  def predict_links(self, x, edge_index_to_predict):
    edge_embeds = torch.cat(
        [x[edge_index_to_predict[0]], x[edge_index_to_predict[1]]], dim=1)
    return self.mlp(edge_embeds).squeeze()


# 📌 Train the model
model = LinkPredictor(in_channels=64, hidden_channels=64)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

print("\n🔄 Training the GraphSAGE model...")
for epoch in range(100):
  optimizer.zero_grad()
  node_embeddings = model(data.x, data.edge_index)
  output = model.predict_links(node_embeddings, train_edge_index)
  loss = criterion(output, train_y)
  loss.backward()
  optimizer.step()
  if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 📌 Define entity extraction patterns for different entity types
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
  'Industry': [
    r'(experience\s+in\s+the\s+[\w\s,\.\-]+(?:industry|field))',
    r'(experience\s+in\s+the\s+computer\s+software)'
  ],
  'Soft_Skill': [
    r'(actively learn|improve from colleagues|willing to contribute|get along with everyone)',
    r'(responsible|dedicated|proactive)'
  ]
}
# Technology categories lookup table
TECHNOLOGY_CATEGORIES = {
  # Programming Languages
  'Python': 'Programming_Language',
  'Java': 'Programming_Language',
  'C++': 'Programming_Language',
  'C': 'Programming_Language',
  'JavaScript': 'Programming_Language',
  'Go': 'Programming_Language',
  'Typescript': 'Programming_Language',
  'Ruby': 'Programming_Language',
  'Scala': 'Programming_Language',
  'Kotlin': 'Programming_Language',

  # Databases
  'SQL': 'Database',
  'MySQL': 'Database',
  'PostgreSQL': 'Database',
  'MongoDB': 'Database',
  'Oracle': 'Database',
  'SQL Server': 'Database',
  'Elasticsearch': 'Database',
  'Redis': 'Database',
  'InfluxDB': 'Database',

  # Tools
  'Jira': 'Tool',
  'Redmine': 'Tool',
  'Grafana': 'Tool',
  'Git': 'Tool',
  'Docker': 'Tool',
  'Kubernetes': 'Tool',
  'Airflow': 'Tool',
  'Flask': 'Tool',
  'Ansible': 'Tool',
  'Prometheus': 'Tool',
  'Nagios': 'Tool',
  'SonarQube': 'Tool',
  'Nexus': 'Tool',

  # Cloud
  'AWS': 'Cloud',
  'Azure': 'Cloud',
  'GCP': 'Cloud',
  'Google Cloud': 'Cloud',
  'cloud platforms': 'Cloud',

  # Automation
  'CI/CD': 'Automation_Tool',
  'Terraform': 'Automation_Tool',
  'ArgoCD': 'Automation_Tool',
  'CloudFormation': 'Automation_Tool',
  'Jenkins': 'Automation_Tool',
  'GitLab CI': 'Automation_Tool',
  'Tekton': 'Automation_Tool',

  # OS
  'Linux': 'OS',
  'Windows': 'OS',
  'macOS': 'OS',
  'Operation System': 'OS',
  'Operating System': 'OS',
  'Ubuntu': 'OS',
  'CentOS': 'OS',

  # Microservices
  'Microservices': 'Microservice',
  'API Gateway': 'Microservice',
  'Service Mesh': 'Microservice',
  'gRPC': 'Microservice',

  # Frameworks
  'Spring': 'Framework',
  'Spring Framework': 'Framework',
  'Spring FrameWork': 'Framework',
  'Spring Boot': 'Framework',
  'Apache Spark': 'Framework',
  'Spark': 'Framework',
  'Hadoop': 'Framework',
  'Django': 'Framework',
  'FastAPI': 'Framework'
}


# 📌 Determine relationship type based on technology category
def determine_technology_relationship(technology_name):
    normalized_name = technology_name.strip().lower()

    # First look in explicit mapping
    for tech, cat in TECHNOLOGY_CATEGORIES.items():
        if normalized_name == tech.lower():
            category = cat
            break
    else:
        # If not found, try pattern matching
        if any(keyword in normalized_name for keyword in ['cloud', 'aws', 'azure', 'gcp']):
            category = 'Cloud'
        elif any(keyword in normalized_name for keyword in ['sql', 'db', 'database', 'postgres', 'mongo', 'redis', 'influx']):
            category = 'Database'
        elif any(keyword in normalized_name for keyword in ['linux', 'windows', 'ubuntu', 'centos', 'os', 'system']):
            category = 'OS'
        elif any(keyword in normalized_name for keyword in ['ci/cd', 'pipeline', 'terraform', 'ansible', 'jenkins', 'argo', 'gitlab']):
            category = 'Automation_Tool'
        elif any(keyword in normalized_name for keyword in ['microservice', 'api gateway', 'grpc', 'service mesh']):
            category = 'Microservice'
        elif any(keyword in normalized_name for keyword in ['jira', 'redmine', 'grafana', 'git', 'docker', 'kubernetes', 'prometheus']):
            category = 'Tool'
        elif any(keyword in normalized_name for keyword in ['spring', 'spark', 'hadoop', 'flask', 'django', 'fastapi']):
            category = 'Framework'
        elif any(keyword in normalized_name for keyword in ['python', 'java', 'c++', 'c#', 'go', 'typescript', 'scala', 'ruby']):
            category = 'Programming_Language'
        else:
            category = 'Technology'  # Default fallback

    # Convert category to relationship type
    relationship_mappings = {
        'Programming_Language': 'PROGRAM_LANGUAGE',
        'Database': 'DATABASE',
        'Tool': 'TOOL',
        'Cloud': 'CLOUD',
        'OS': 'OS',
        'Automation_Tool': 'AUTO_TOOL',
        'Microservice': 'MICROSERVICE',
        'Framework': 'FRAMEWORK',
        'Technology': 'HAS_TECHNOLOGY'  # Default relationship
    }

    return relationship_mappings.get(category, 'HAS_TECHNOLOGY')


# 📌 Improved entity extraction function
def extract_entities(about_text, entity_types=None):
  """Extract entities from About text based on specified entity types."""
  if entity_types is None:
    entity_types = ENTITY_PATTERNS.keys()

  entities = {entity_type: [] for entity_type in entity_types}

  # Apply patterns for each requested entity type
  for entity_type in entity_types:
    if entity_type in ENTITY_PATTERNS:
      for pattern in ENTITY_PATTERNS[entity_type]:
        matches = re.findall(pattern, about_text, re.IGNORECASE)
        entities[entity_type].extend([match.strip() for match in matches])

  # Clean up entities (remove duplicates, truncate long entries)
  for entity_type in entity_types:
    entities[entity_type] = list(set(entities[entity_type]))
    # Truncate long entities
    entities[entity_type] = [entity[:100] if len(entity) > 100 else entity
                             for entity in entities[entity_type]]

  return entities


# 📌 Improved confidence calculation
def calculate_confidence(prediction_score, entity_text, about_text,
    entity_type):
  """Calculate confidence for an extracted entity."""
  # Weight parameters
  alpha = 0.7  # Weight for GraphSAGE prediction
  beta = 0.2  # Weight for entity specificity
  gamma = 0.1  # Weight for context quality

  # Measure entity specificity (longer and more detailed is better)
  specificity_score = min(1.0, len(entity_text.split()) / 10)

  # Default context score
  context_score = 0.0

  # Find entity position in the about text
  entity_pos = about_text.lower().find(entity_text.lower())
  if entity_pos >= 0:
    # Get context around the entity
    start = max(0, entity_pos - 30)
    end = min(len(about_text), entity_pos + len(entity_text) + 30)
    context = about_text[start:end]

    # Context keywords based on entity type
    context_keywords = {
      'Skill': ["experience", "expert", "skill", "proficient", "knowledge",
                "know how"],
      'Programming_Language': ["Python", "Java", "C++", "JavaScript", "SQL",
                               "code", "develop", "program"],
      'Tool': ["tool", "use", "using", "Jira", "Redmine", "Grafana", "manage"],
      'Automation_Tool': ["automation", "CI/CD", "pipeline", "deploy",
                          "terraform", "kubernetes"],
      'Microservice': ["microservice", "service", "API", "REST", "gRPC"],
      'OS': ["OS", "system", "platform", "Linux", "Windows",
             "Operation System"],
      'Database': ["database", "DB", "data", "SQL", "NoSQL", "query"],
      'Cloud': ["cloud", "AWS", "Azure", "GCP", "service"],
      'Experience_Level': ["years", "year", "experience", "worked", "+"],
      'Project': ["project", "worked on", "involved in", "developed"],
      'Seniority': ["years", "year", "senior", "junior", "lead"]
    }

    # Use appropriate keywords or fallback to general ones
    keywords = context_keywords.get(entity_type,
                                    ["experience", "skill", "worked",
                                     "knowledge", "years"])

    # Count keyword occurrences in context
    keyword_count = sum(
        1 for keyword in keywords if keyword.lower() in context.lower())
    context_score = min(1.0, keyword_count / 3)

  # Calculate total confidence
  confidence = alpha * prediction_score + beta * specificity_score + gamma * context_score

  return confidence, {
    "prediction": prediction_score,
    "specificity": specificity_score,
    "context": context_score
  }


# 📌 Determine relationship type based on entity types
def determine_relationship_type(source_type, target_type, target_value=None):
  """Determine relationship type based on source and target node types."""
  if source_type == 'Employee':
    if target_type == 'Skill':
      return 'HAS_DETAIL_SKILL'
    elif target_type == 'Technology':
      # Use specialized function to determine technology relationship
      if target_value:
        return determine_technology_relationship(target_value)
      return 'HAS_TECHNOLOGY'  # Default if no target value
    elif target_type == 'Experience_Level':
      return 'HAS_EXPERIENCE_LEVEL'
    elif target_type == 'Project':
      return 'WORKED_ON'
    elif target_type == 'Seniority':
      return 'HAS_SENIORITY'
    elif target_type == 'Industry':
      return 'HAS_INDUSTRY_EXPERIENCE'
    elif target_type == 'Soft_Skill':
      return 'HAS_SOFT_SKILL'

  # Default case
  return 'RELATED_TO'


# 📌 Process About nodes and extract entities
print("\n🔍 Processing About nodes and extracting entities...")
employees_with_about = result[result['node_type'] == 'About']

for _, row in employees_with_about.iterrows():
  employee_id = row['source']
  about_id = row['target']
  employee_name = row['employee_name']
  about_text = row['node_name']

  # Skip if about text is empty
  if about_text is None or about_text.strip() == "":
    print(f"⚠ {employee_name} has empty About text (SKIPPING)")
    continue

  print(f"\n📑 Processing About text for {employee_name}:")

  # Extract entities from About text
  entity_types = ['Skill', 'Programming_Language', 'Tool', 'Automation_Tool',
                'Microservice', 'OS', 'Database', 'Cloud', 'Experience_Level',
                'Project', 'Seniority']
  extracted_entities = extract_entities(about_text, entity_types)

  # Get node embeddings for confidence calculation
  with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index)

  # Process each entity type
  new_relationships = []

  for entity_type, entities in extracted_entities.items():
    if not entities:
      continue

    print(f"  Found {len(entities)} {entity_type} entities")

    for entity in entities:
      # Calculate prediction score (using employee embedding)
      # This is simplified - in a real scenario we would create a temporary node
      prediction_score = 0.75  # Default confidence from graph structure

      # Calculate overall confidence
      confidence, details = calculate_confidence(
          prediction_score, entity, about_text, entity_type
      )

      # Only keep relationships with confidence above threshold
      if confidence > 0.6:
        relationship_type = determine_relationship_type('Employee', entity_type)
        new_relationships.append({
          'employee_name': employee_name,
          'entity': entity,
          'entity_type': entity_type,
          'relationship_type': relationship_type,
          'confidence': confidence
        })

        print(
          f"  🔄 {entity_type}: {entity[:50]}{'...' if len(entity) > 50 else ''} [{confidence:.2f}]")

  # Add relationships to Neo4j
  for rel in new_relationships:
    # Create Cypher query to add the relationship
    cypher_query = f"""
          MATCH (e:Employee {{name: $employee_name}})
          MERGE (t:{rel['entity_type']} {{name: $entity_name}})
          MERGE (e)-[r:{rel['relationship_type']}]->(t)
          SET r.confidence = $confidence,
              r.extracted_from = 'about_text'
          """

    # Execute the query
    try:
      graph.run(
          cypher_query,
          employee_name=rel['employee_name'],
          entity_name=rel['entity'],
          confidence=rel['confidence']
      )
      print(
          f"  ✅ Added: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}...")
    except Exception as e:
      print(f"  ❌ Error adding relationship: {e}")

print("\n🎯 Processing complete!")