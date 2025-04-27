"""
Utility functions for the knowledge graph extraction system.
"""

import re
import torch
import numpy as np
from neo4j import GraphDatabase


def format_entity_for_display(entity, max_length=50):
    """
    Format entity text for display in console output.

    Args:
        entity: The entity text to format
        max_length: Maximum length before truncating

    Returns:
        Formatted entity text with ellipsis if needed
    """
    if len(entity) > max_length:
        return f"{entity[:max_length]}..."
    return entity


def extract_years_from_experience(experience_text):
    """
    Extract the number of years from an experience description.

    Args:
        experience_text: Text describing experience

    Returns:
        Integer representing years of experience, or None if not found
    """
    pattern = r'(\d+)\+?\s*years?'
    match = re.search(pattern, experience_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two text strings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Jaccard similarity score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0


def find_context_window(text, entity, window_size=30):
    """
    Extract context window around an entity in text.

    Args:
        text: Full text to search in
        entity: Entity to find context for
        window_size: Number of characters before and after entity

    Returns:
        Context window as string, or empty string if entity not found
    """
    entity_pos = text.lower().find(entity.lower())
    if entity_pos < 0:
        return ""

    start = max(0, entity_pos - window_size)
    end = min(len(text), entity_pos + len(entity) + window_size)

    return text[start:end]


def create_weighted_embedding(embeddings, weights):
    """
    Create a weighted average of embeddings.

    Args:
        embeddings: List of embedding tensors
        weights: List of weights for each embedding

    Returns:
        Weighted average embedding tensor
    """
    if not embeddings or sum(weights) == 0:
        return None

    weighted_sum = torch.zeros_like(embeddings[0])
    for emb, weight in zip(embeddings, weights):
        weighted_sum += emb * weight

    return weighted_sum / sum(weights)


def log_relationship_creation(employee, relationship_type, entity_type, entity, confidence):
    """
    Format a log message for relationship creation.

    Args:
        employee: Employee name
        relationship_type: Type of relationship
        entity_type: Type of entity
        entity: Entity name
        confidence: Confidence score

    Returns:
        Formatted log message
    """
    return f"{employee} -[:{relationship_type} ({confidence:.2f})]-> {entity_type}:{format_entity_for_display(entity, 30)}"


def sanitize_cypher_string(text):
    """
    Sanitize a string for use in Cypher queries.

    Args:
        text: String to sanitize

    Returns:
        Sanitized string
    """
    # Replace single quotes with escaped single quotes
    return text.replace("'", "\\'")


def generate_node_creation_query(label, properties):
    """
    Generate a Cypher query to create a node.

    Args:
        label: Node label
        properties: Dictionary of node properties

    Returns:
        Cypher query string
    """
    props_str = ", ".join([f"{k}: '{sanitize_cypher_string(str(v))}'"
                           for k, v in properties.items()])

    return f"CREATE (:{label} {{{props_str}}})"


def generate_relationship_creation_query(from_label, from_props, rel_type,
                                         to_label, to_props, rel_props=None):
    """
    Generate a Cypher query to create a relationship.

    Args:
        from_label: Label of source node
        from_props: Dictionary of source node properties
        rel_type: Relationship type
        to_label: Label of target node
        to_props: Dictionary of target node properties
        rel_props: Dictionary of relationship properties

    Returns:
        Cypher query string
    """
    from_props_str = ", ".join([f"{k}: '{sanitize_cypher_string(str(v))}'"
                                for k, v in from_props.items()])

    to_props_str = ", ".join([f"{k}: '{sanitize_cypher_string(str(v))}'"
                              for k, v in to_props.items()])

    rel_props_str = ""
    if rel_props:
        rel_props_str = " {" + ", ".join([f"{k}: '{sanitize_cypher_string(str(v))}'"
                                          for k, v in rel_props.items()]) + "}"

    return f"""
    MATCH (a:{from_label} {{{from_props_str}}})
    MATCH (b:{to_label} {{{to_props_str}}})
    CREATE (a)-[:{rel_type}{rel_props_str}]->(b)
    """