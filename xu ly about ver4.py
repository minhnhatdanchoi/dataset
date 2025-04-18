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
WHERE type(r) <> 'HAS_ABOUT' AND any(label IN labels(n) WHERE label IN ['Nationality', 'Language', 'Project', 'Technology', 'Seniority'])
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
    self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=2)
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
    r'\b(Redmine|Grafana|Jira)(?:[,\s]+(?:and\s+)?(Redmine|Grafana|Jira))*\b'
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

# 📌 Improved entity extraction function
def extract_entities(about_text, entity_types=None):
  """Extract entities from About text based on specified entity types."""
  if entity_types is None:
    entity_types = ENTITY_PATTERNS.keys()

  entities = {entity_type: [] for entity_type in entity_types}

  # Process sentences separately for better context
  sentences = re.split(r'[.;]',
                       about_text)  # Split by periods and semicolons only
  sentences = [s.strip() for s in sentences if s.strip()]

  # Apply patterns for each sentence and requested entity type
  for sentence in sentences:
    for entity_type in entity_types:
      if entity_type in ENTITY_PATTERNS:
        for pattern in ENTITY_PATTERNS[entity_type]:
          matches = re.findall(pattern, sentence, re.IGNORECASE)
          if matches:
            # Extract individual entities from comma-separated lists
            for match in matches:
              if isinstance(match, tuple):  # For regex groups
                match = match[0]  # Take the first group

              # Handle comma-separated lists of entities
              if ',' in match:
                # If it's a comma-separated list, split and process each item
                split_entities = [m.strip() for m in re.split(r',\s*', match)]
                for split_entity in split_entities:
                  # Re-check each split entity against the pattern
                  if re.search(pattern, split_entity, re.IGNORECASE):
                    entities[entity_type].append(split_entity)
              else:
                entities[entity_type].append(match.strip())

  # Special processing for Experience_Level to extract duration and associated technology
  if 'Experience_Level' in entity_types and entities['Experience_Level']:
    refined_experiences = []
    for exp in entities['Experience_Level']:
      # Extract years
      duration_pattern = r'(\d+\+?)\s*years?'
      duration_match = re.search(duration_pattern, exp, re.IGNORECASE)
      if duration_match:
        duration = duration_match.group(1)

        # Try to extract technology/field the experience relates to
        tech_pattern = r'experience\s+(?:in|with|of)\s+([\w\s,\.\-]+)'
        tech_match = re.search(tech_pattern, exp, re.IGNORECASE)

        # If no match with standard pattern, try alternate pattern
        if not tech_match:
          tech_pattern = r'experience\s+(?:in|with|of)\s+the\s+([\w\s,\.\-]+)'
          tech_match = re.search(tech_pattern, exp, re.IGNORECASE)

        if tech_match:
          tech = tech_match.group(1).strip()
          # Clean up the technology string
          tech = re.sub(r'[,\.]$', '', tech).strip()
          refined_experiences.append(f"{duration} years in {tech}")
        else:
          refined_experiences.append(f"{duration} years")

    entities['Experience_Level'] = refined_experiences

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
  if entity_type == 'Technology':
    alpha = 0.65  # Weight for GraphSAGE prediction
    beta = 0.2  # Weight for entity specificity
    gamma = 0.15  # Weight for context quality
  elif entity_type in ['Skill', 'Soft_Skill']:
    alpha = 0.4  # Reduce prediction weight for Skill/Soft_Skill
    beta = 0.3  # Increase specificity weight
    gamma = 0.3  # Increase context weight
  elif entity_type == 'Experience_Level':
    alpha = 0.5  # More balanced for Experience_Level
    beta = 0.25
    gamma = 0.25
  else:
    alpha = 0.6  # Default weights
    beta = 0.25
    gamma = 0.15
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
                "know how", "skilled", "responsible", "dedicated"],
      'Technology': ["Python", "Java", "C++", "JavaScript", "SQL","code", "develop", "program", "C","tool", "use", "using", "Jira", "Redmine", "Grafana", "manage","automation", "CI/CD", "pipeline", "deploy","terraform", "kubernetes", "ArgoCD","microservice", "service", "API", "REST", "gRPC","OS", "system", "platform", "Linux", "Windows","Operation System","database", "DB", "data", "SQL", "NoSQL", "query","cloud", "AWS", "Azure", "GCP", "service", "infrastructure","resources","spring", "framework", "hadoop", "spark", "apache","platform", "bidata", "big data"],
      'Experience_Level': ["years", "year", "experience", "worked", "+"],
      'Project': ["project", "worked on", "involved in", "developed"],
      'Soft_Skill': ["learn", "improve", "contribute", "proactive",
                     "responsible", "dedicated", "get along"]
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
# 5. Thêm ngưỡng confidence riêng cho từng loại thực thể
CONFIDENCE_THRESHOLDS = {
    'Technology': 0.6,
    'Skill': 0.4,         # Giảm ngưỡng cho Skill
    'Soft_Skill': 0.4,    # Giảm ngưỡng cho Soft_Skill
    'Experience_Level': 0.45,
    'Project': 0.5
}

# 📌 Determine relationship type based on entity types
def determine_relationship_type(source_type, target_type):
    """Determine relationship type based on source and target node types."""
    if source_type == 'Employee':
        if target_type == 'Skill':
            return 'HAS_DETAIL_SKILL'
        elif target_type == 'Technology':
            return 'HAS_TECHNOLOGY'
        elif target_type == 'Experience_Level':
            return 'HAS_EXPERIENCE_LEVEL'
        elif target_type == 'Project':
            return 'WORKED_ON'
        elif target_type == 'Industry':
            return 'HAS_INDUSTRY_EXPERIENCE'
        elif target_type == 'Soft_Skill':
            return 'HAS_SOFT_SKILL'

    # Default case
    return 'RELATED_TO'


# Add this function to create temporary node embedding for new entities
def create_temp_entity_embedding(entity_text, entity_type, node_embeddings):
  """
  Create a temporary embedding for a new entity based on similar entities in the graph.

  Args:
      entity_text: The text of the new entity
      entity_type: The type of the entity (Technology, Skill, etc.)
      node_embeddings: The current node embeddings from the model

  Returns:
      A tensor embedding for the new entity
  """
  # Find existing nodes of the same type to use as reference
  similar_nodes = []

  # Get IDs of existing nodes with the same type
  for node_id, info in id_to_info.items():
    if info['type'] == entity_type:
      # Calculate text similarity (simple word overlap for now)
      entity_words = set(entity_text.lower().split())
      node_words = set(info['name'].lower().split())

      # Calculate Jaccard similarity
      if len(entity_words) > 0 and len(node_words) > 0:
        intersection = len(entity_words.intersection(node_words))
        union = len(entity_words.union(node_words))
        similarity = intersection / union if union > 0 else 0

        similar_nodes.append((node_id, similarity))

  # Sort by similarity (highest first)
  similar_nodes.sort(key=lambda x: x[1], reverse=True)

  # If we have similar nodes, use weighted average of top 3 as embedding
  if similar_nodes:
    # Take top 3 or as many as available
    top_nodes = similar_nodes[:min(3, len(similar_nodes))]

    # If no similar nodes have similarity > 0, use random embedding
    if sum(sim for _, sim in top_nodes) == 0:
      return torch.randn(node_embeddings.shape[1])

    # Calculate weighted average embedding
    weighted_sum = torch.zeros(node_embeddings.shape[1])
    total_weight = 0

    for node_id, similarity in top_nodes:
      if similarity > 0:
        weighted_sum += node_embeddings[node_id] * similarity
        total_weight += similarity

    if total_weight > 0:
      return weighted_sum / total_weight

  # Fallback: Return a random embedding biased towards the average of all entities of this type
  type_embeddings = [node_embeddings[node_id] for node_id, info in
                     id_to_info.items()
                     if info['type'] == entity_type]

  if type_embeddings:
    # Return random embedding biased towards the average of this type
    avg_embedding = torch.stack(type_embeddings).mean(dim=0)
    random_factor = 0.3
    return avg_embedding * (1 - random_factor) + torch.randn_like(
      avg_embedding) * random_factor

  # Last resort: completely random embedding
  return torch.randn(node_embeddings.shape[1])


# Now modify the Process About nodes section to use the trained model for prediction
# Replace the "Process About nodes and extract entities" section with this improved code:
# 📌 Process About nodes and extract entities
print("\n🔍 Processing About nodes and extracting entities...")
employees_with_about = result[result['node_type'] == 'About']

# Get node embeddings from trained model
with torch.no_grad():
  node_embeddings = model(data.x, data.edge_index)

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
  entity_types = ['Skill', 'Technology', 'Experience_Level', 'Project',
                  'Soft_Skill']
  extracted_entities = extract_entities(about_text, entity_types)

  # Process each entity type
  new_relationships = []

  for entity_type, entities in extracted_entities.items():
    if not entities:
      continue

    print(f"  Found {len(entities)} {entity_type} entities")

    for entity in entities:
      # Create temporary embedding for the new entity
      temp_entity_embedding = create_temp_entity_embedding(entity, entity_type,
                                                           node_embeddings)

      # Get employee embedding
      employee_embedding = node_embeddings[employee_id]

      # Calculate prediction score using the trained model
      # We'll concatenate employee and entity embeddings
      edge_embedding = torch.cat([employee_embedding, temp_entity_embedding],
                                 dim=0).unsqueeze(0)

      # Use model to predict link probability
      with torch.no_grad():
        prediction_score_tensor = torch.sigmoid(
          model.mlp(edge_embedding).squeeze())
        prediction_score = prediction_score_tensor.item()

      # Calculate overall confidence
      confidence, details = calculate_confidence(
          prediction_score, entity, about_text, entity_type
      )



      # Debug
      #print(
        #f"  DEBUG {entity_type}: {entity[:30]} - Confidence: {confidence:.2f}, Prediction: {prediction_score:.2f},Specificity: {details['specificity']:.2f}, Context: {details['context']:.2f}")

      threshold = CONFIDENCE_THRESHOLDS.get(entity_type, 0.6)
      # Only keep relationships with confidence above threshold
      if confidence > threshold:
        relationship_type = determine_relationship_type('Employee', entity_type)
        new_relationships.append({
          'employee_name': employee_name,
          'entity': entity,
          'entity_type': entity_type,
          'relationship_type': relationship_type,
          'confidence': confidence,
          'prediction_score': prediction_score,
          'context_score': details['context'],
          'specificity_score': details['specificity']
        })

        print(
          f"  🔄 {entity_type}: {entity[:50]}{'...' if len(entity) > 50 else ''} [Conf: {confidence:.2f}, Pred: {prediction_score:.2f}]")

  # Add relationships to Neo4j
  for rel in new_relationships:
    # Create Cypher query to add the relationship
    cypher_query = f"""
            MATCH (e:Employee {{name: $employee_name}})
            MERGE (t:{rel['entity_type']} {{name: $entity_name}})
            MERGE (e)-[r:{rel['relationship_type']}]->(t)
            SET r.confidence = $confidence,
                r.prediction_score = $prediction_score,
                r.context_score = $context_score,
                r.specificity_score = $specificity_score,
                r.extracted_from = 'about_text'
            """

    # Execute the query
    try:
      graph.run(
          cypher_query,
          employee_name=rel['employee_name'],
          entity_name=rel['entity'],
          confidence=rel['confidence'],
          prediction_score=rel['prediction_score'],
          context_score=rel['context_score'],
          specificity_score=rel['specificity_score']
      )
      print(
          f"  ✅ Added: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}..."
      )
    except Exception as e:
      print(f"  ❌ Error adding relationship: {e}")

print("\n🎯 Processing complete!")