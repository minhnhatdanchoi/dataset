from py2neo import Graph
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

# 🔗 Connect to Neo4j
NEO4J_URI = "neo4j+s://890a15f5.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "NV_XHqxDtbfaxIrqbRTlJCXjUwQSipP1nN1r60VHHhw"

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
x = torch.randn(n_nodes, 128)  # Random feature vector for each node
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
    self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=100)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_channels * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

  def forward(self, x, edge_index):
    x = self.sage(x, edge_index)
    return x

  def predict_links(self, x, edge_index_to_predict):
    edge_embeds = torch.cat(
        [x[edge_index_to_predict[0]], x[edge_index_to_predict[1]]], dim=1)
    return self.mlp(edge_embeds).squeeze()


# 📌 Train the model
model = LinkPredictor(in_channels=128, hidden_channels=256)
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
    r'(\d+\+?\s+years?\s+(?:of\s+)?experience\s+(?:in|with)\s+[\w\s,\.]+)',
    r'(proficient\s+(?:in|with)\s+[\w\s,\.]+)',
    r'(skilled\s+(?:in|with)\s+[\w\s,\.]+)',
    r'(knowledge\s+of\s+[\w\s,\.]+)'
  ],
  'Technology': [
    r'(experience\s+(?:in|with)\s+[\w\s,\.#]+)',
    r'(worked\s+with\s+[\w\s,\.#]+)',
    r'(familiar\s+with\s+[\w\s,\.#]+)',
    r'([\w\s,\.#]+\s+developer)',
    r'([\w\s,\.#]+\s+programming)'
  ],
  'Project': [
    r'(worked\s+on\s+[\w\s,\.]+\s+project)',
    r'(developed\s+[\w\s,\.]+)',
    r'(implemented\s+[\w\s,\.]+)',
    r'(project\s+(?:called|named)\s+[\w\s,\.]+)'
  ],
  'Experience': [
    r'(worked\s+(?:at|for)\s+[\w\s,\.]+)',
    r'(position\s+(?:as|of)\s+[\w\s,\.]+)',
    r'([\w\s,\.]+\s+role\s+at\s+[\w\s,\.]+)'
  ],
  'Seniority': [
    r'(senior\s+[\w\s,\.]+)',
    r'(junior\s+[\w\s,\.]+)',
    r'(lead\s+[\w\s,\.]+)',
    r'([\w\s,\.]+\s+with\s+\d+\+?\s+years\s+of\s+experience)',
    r'\b\d+\s+years?\b',
    r'\b\d+\s+year?\b'
  ]
}


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
      'Skill': ["experience", "expert", "skill", "proficient", "knowledge"],
      'Experience': ["worked", "position", "role", "job", "company"],
      'Project': ["project", "worked on", "involved in", "developed"],
      'Technology': ["technology", "familiar", "worked with", "experience in",
                     "knowledge","Python", "Java", "C++", "JavaScript", "SQL","AWS", "Azure", "GCP", "Docker", "Kubernetes","Windows", "Linux", "macOS","HTML", "CSS", "React", "Angular", "Node.js","Ruby", "PHP", "Swift", "Go","TensorFlow", "PyTorch", "scikit-learn","Hadoop", "Spark", "Kafka","MongoDB", "PostgreSQL", "MySQL"],
      'Seniority': ["years", "year", "senior", "junior", "lead","1", "2", "3", "4", "5","6", "7", "8", "9", "10", "11", "12", "13", "14", "15","16", "17", "18", "19", "20"]
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
def determine_relationship_type(source_type, target_type):
  """Determine relationship type based on source and target node types."""
  if source_type == 'Employee':
    if target_type == 'Skill':
      return 'HAS_DETAIL_SKILL'
    elif target_type == 'Technology':
      return 'DETAIL_EX_TECHNOLOGY'
    elif target_type == 'Experience':
      return 'WORKED_AS'
    elif target_type == 'Seniority':
      return 'HAS_SENIORITY_TECHNOLOGY'
    elif target_type == 'Project':
      return 'WORKED_ON'

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
  entity_types = ['Skill', 'Technology', 'Project', 'Experience', 'Seniority']
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