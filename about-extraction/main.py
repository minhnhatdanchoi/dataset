"""
Main module for extracting entities from employee profiles and building a knowledge graph.
This script uses GraphSAGE for graph representation learning and link prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from py2neo import Graph
from neo4j import GraphDatabase
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

# Import configuration and data structures from separate files
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    MODEL_PARAMS, TRAINING_PARAMS,
    CONFIDENCE_THRESHOLDS, CONFIDENCE_WEIGHTS
)
from entity_patterns import ENTITY_PATTERNS
from context_keywords import CONTEXT_KEYWORDS

# ============================
# Neo4j Connection
# ============================
def connect_to_neo4j():
    """Establish connection to Neo4j database."""
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("✅ Neo4j connection successful!")
        return graph
    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        exit()

# ============================
# Data Extraction
# ============================
def extract_data_from_neo4j(graph):
    """Extract node and relationship data from Neo4j."""
    # Query to extract existing relationships
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

    return result

def extract_existing_technology_relationships(graph):
    """Extract existing relationships between employees and technology nodes."""
    query = """
    MATCH (e:Employee)-[r]->(t:Technology)
    RETURN e.name AS employee_name, t.name AS technology_name, type(r) AS relationship_type
    """
    existing_tech_relationships = graph.run(query).to_data_frame()

    # Create a dictionary to track existing technology relationships for each employee
    employee_tech_relationships = {}
    if not existing_tech_relationships.empty:
        for _, row in existing_tech_relationships.iterrows():
            employee = row['employee_name']
            tech = row['technology_name']
            if employee not in employee_tech_relationships:
                employee_tech_relationships[employee] = set()
            employee_tech_relationships[employee].add(tech.lower())

        print(f"Found existing technology relationships for {len(employee_tech_relationships)} employees")
    else:
        print("No existing technology relationships found")

    return employee_tech_relationships

# ============================
# Graph Construction
# ============================
def prepare_graph_data(result):
    """Prepare graph data for model training."""
    # Convert data to tensors
    edge_index = torch.tensor(result[['source', 'target']].values,
                              dtype=torch.long).t().contiguous()
    n_nodes = max(edge_index.flatten()).item() + 1
    x = torch.randn(n_nodes, 64)  # Random feature vector for each node
    data = Data(x=x, edge_index=edge_index)

    # Create ID to entity mapping dictionaries
    id_to_info = {}
    for _, row in result.iterrows():
        id_to_info[row['source']] = {'name': row['employee_name'], 'type': 'Employee'}
        id_to_info[row['target']] = {'name': row['node_name'],
                                     'type': row['node_type']}

    # Generate negative samples
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

    return data, train_edge_index, y, id_to_info

# ============================
# Model Definition
# ============================
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2):
        """
        Initialize the LinkPredictor model.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
        """
        super().__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=num_layers, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        """Generate node embeddings."""
        x = self.sage(x, edge_index)
        return x

    def predict_links(self, x, edge_index_to_predict):
        """Predict links between nodes."""
        edge_embeds = torch.cat(
            [x[edge_index_to_predict[0]], x[edge_index_to_predict[1]]], dim=1)
        return self.mlp(edge_embeds).squeeze()

# ============================
# Model Training
# ============================
def train_model(model, data, train_edge_index, train_y, epochs=100, lr=0.01, print_interval=10):
    """Train the LinkPredictor model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("\n🔄 Training the GraphSAGE model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        node_embeddings = model(data.x, data.edge_index)
        output = model.predict_links(node_embeddings, train_edge_index)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        if epoch % print_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

# ============================
# Entity Extraction
# ============================
def extract_entities(about_text, entity_types=None):
    """
    Extract entities from About text based on specified entity types.

    Args:
        about_text: Text to extract entities from
        entity_types: List of entity types to extract

    Returns:
        Dictionary of extracted entities by type
    """
    if entity_types is None:
        entity_types = ENTITY_PATTERNS.keys()

    entities = {entity_type: [] for entity_type in entity_types}

    # Process sentences separately for better context
    sentences = re.split(r'[.;]', about_text)  # Split by periods and semicolons only
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

def calculate_confidence(prediction_score, entity_text, about_text, entity_type):
    """
    Calculate confidence score for an extracted entity based on multiple factors.

    Args:
        prediction_score: Score from the GraphSAGE model
        entity_text: The extracted entity text
        about_text: Original text the entity was extracted from
        entity_type: Type of the entity

    Returns:
        Tuple of (confidence_score, details_dict)
    """
    # Get weights based on entity type
    weights = CONFIDENCE_WEIGHTS.get(entity_type, CONFIDENCE_WEIGHTS['default'])
    alpha = weights['prediction']  # Weight for GraphSAGE prediction
    beta = weights['specificity']  # Weight for entity specificity
    gamma = weights['context']     # Weight for context quality

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

        # Get appropriate keywords for this entity type
        keywords = CONTEXT_KEYWORDS.get(entity_type,
                                       ["experience", "skill", "worked", "knowledge", "years"])

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

def determine_relationship_type(source_type, target_type):
    """
    Determine relationship type based on source and target node types.

    Args:
        source_type: Type of the source node
        target_type: Type of the target node

    Returns:
        String representing the relationship type
    """
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

def create_temp_entity_embedding(entity_text, entity_type, node_embeddings, id_to_info):
    """
    Create a temporary embedding for a new entity based on similar entities in the graph.

    Args:
        entity_text: The text of the new entity
        entity_type: The type of the entity (Technology, Skill, etc.)
        node_embeddings: The current node embeddings from the model
        id_to_info: Dictionary mapping node IDs to node information

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

def get_existing_technology_node(graph, technology_name):
    """
    Check if a technology node with the given name exists in the database.

    Args:
        graph: Neo4j graph connection
        technology_name: The name of the technology to check

    Returns:
        Boolean indicating if the node exists
    """
    query = """
    MATCH (t:Technology {name: $tech_name})
    RETURN count(t) > 0 AS exists
    """
    result = graph.evaluate(query, tech_name=technology_name)
    return result

# ============================
# Main Processing Function
# ============================
def process_about_texts(graph, result, node_embeddings, model, id_to_info, employee_tech_relationships):
    """
    Process About nodes and extract entities.

    Args:
        graph: Neo4j graph connection
        result: DataFrame containing graph data
        node_embeddings: Node embeddings from the trained model
        model: Trained LinkPredictor model
        id_to_info: Dictionary mapping node IDs to node information
        employee_tech_relationships: Dictionary tracking existing technology relationships

    Returns:
        Statistics about extracted entities and relationships
    """
    print("\n🔍 Processing About nodes and extracting entities...")
    employees_with_about = result[result['node_type'] == 'About']

    # Counters for summary statistics
    total_hidden_nodes_found = 0
    total_hidden_relationships_found = 0
    hidden_nodes_by_type = {
        'Technology': 0,
        'Skill': 0,
        'Soft_Skill': 0,
        'Experience_Level': 0,
        'Project': 0
    }
    hidden_relationships_by_type = {
        'HAS_TECHNOLOGY': 0,
        'HAS_DETAIL_SKILL': 0,
        'HAS_SOFT_SKILL': 0,
        'HAS_EXPERIENCE_LEVEL': 0,
        'WORKED_ON': 0
    }

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
                # Special handling for Technology entities - CHECK FOR EXISTING RELATIONSHIPS
                if entity_type == 'Technology':
                    # Convert to lowercase for case-insensitive comparison
                    entity_lower = entity.lower()

                    # Check if the employee already has a relationship to this technology (regardless of relationship type)
                    if employee_name in employee_tech_relationships and entity_lower in employee_tech_relationships[
                        employee_name]:
                        print(f"  ⏩ Skipping {entity}: {employee_name} already has a relationship to this technology")
                        continue

                # Create temporary embedding for the new entity
                temp_entity_embedding = create_temp_entity_embedding(entity, entity_type,
                                                                     node_embeddings, id_to_info)

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
            # For Technology entities, first check if the node already exists
            node_exists = False
            if rel['entity_type'] == 'Technology':
                node_exists = get_existing_technology_node(graph, rel['entity'])
            else:
                # Check if the node exists for other entity types
                check_node_query = f"""
          MATCH (n:{rel['entity_type']} {{name: $name}})
          RETURN count(n) > 0 AS exists
          """
                node_exists = graph.evaluate(check_node_query, name=rel['entity'])

            # Create Cypher query to add the relationship
            # Use MERGE for both the node and relationship to ensure they exist without duplicating
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

                # Update statistics counters
                total_hidden_relationships_found += 1
                hidden_relationships_by_type[rel['relationship_type']] = hidden_relationships_by_type.get(
                    rel['relationship_type'], 0) + 1

                if not node_exists:
                    total_hidden_nodes_found += 1
                    hidden_nodes_by_type[rel['entity_type']] = hidden_nodes_by_type.get(rel['entity_type'], 0) + 1

                # Different message depending on whether the node existed before
                if node_exists:
                    print(
                        f"  ✅ Added relationship to existing node: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}..."
                    )
                else:
                    print(
                        f"  ✅ Added new node and relationship: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}..."
                    )

                # If this is a Technology entity, update our in-memory tracking
                if rel['entity_type'] == 'Technology':
                    if rel['employee_name'] not in employee_tech_relationships:
                        employee_tech_relationships[rel['employee_name']] = set()
                    employee_tech_relationships[rel['employee_name']].add(rel['entity'].lower())

            except Exception as e:
                print(f"  ❌ Error adding relationship: {e}")

    # Return statistics
    return {
        'total_hidden_nodes': total_hidden_nodes_found,
        'total_hidden_relationships': total_hidden_relationships_found,
        'hidden_nodes_by_type': hidden_nodes_by_type,
        'hidden_relationships_by_type': hidden_relationships_by_type
    }

# ============================
# Print Summary Statistics
# ============================
def print_summary_statistics(stats):
    """Print summary statistics of the extraction process."""
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Total hidden nodes discovered: {stats['total_hidden_nodes']}")
    print(f"Total hidden relationships discovered: {stats['total_hidden_relationships']}")

    if stats['total_hidden_nodes'] > 0:
        print("\nHidden nodes by type:")
        for node_type, count in stats['hidden_nodes_by_type'].items():
            if count > 0:
                print(f"  - {node_type}: {count}")

    if stats['total_hidden_relationships'] > 0:
        print("\nHidden relationships by type:")
        for rel_type, count in stats['hidden_relationships_by_type'].items():
            if count > 0:
                print(f"  - {rel_type}: {count}")

# ============================
# Main Execution
# ============================
def main():
    """Main execution function."""
    # Connect to Neo4j
    graph = connect_to_neo4j()

    # Extract data
    result = extract_data_from_neo4j(graph)
    employee_tech_relationships = extract_existing_technology_relationships(graph)

    # Prepare graph data
    data, train_edge_index, train_y, id_to_info = prepare_graph_data(result)

    # Initialize and train model
    model = LinkPredictor(
        in_channels=MODEL_PARAMS['in_channels'],
        hidden_channels=MODEL_PARAMS['hidden_channels'],
        num_layers=MODEL_PARAMS['num_layers'],
        dropout=MODEL_PARAMS['dropout']
    )

    model = train_model(
        model, data, train_edge_index, train_y,
        epochs=TRAINING_PARAMS['epochs'],
        lr=TRAINING_PARAMS['learning_rate'],
        print_interval=TRAINING_PARAMS['print_interval']
    )

    # Get node embeddings from trained model
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index)

    # Process About texts
    stats = process_about_texts(
        graph, result, node_embeddings, model,
        id_to_info, employee_tech_relationships
    )

    # Print summary statistics
    print_summary_statistics(stats)

    print("\n🎯 Processing complete!")

if __name__ == "__main__":
    main()