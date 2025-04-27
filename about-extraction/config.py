"""
Configuration parameters for the knowledge graph extraction system
"""

# Neo4j connection parameters
NEO4J_URI = "neo4j+s://2413b494.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ecPHA_8U14w5h1AXLs2OtEX4gOh3lrHm7Trl6J-k0Bc"

# Model parameters
MODEL_PARAMS = {
    'in_channels': 64,
    'hidden_channels': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# Training parameters
TRAINING_PARAMS = {
    'learning_rate': 0.01,
    'epochs': 100,
    'print_interval': 10
}

# Confidence thresholds for entity extraction
CONFIDENCE_THRESHOLDS = {
    'Technology': 0.6,
    'Skill': 0.4,  # Lower threshold for skills
    'Soft_Skill': 0.4,  # Lower threshold for soft skills
    'Experience_Level': 0.45,
    'Project': 0.5
}

# Confidence calculation weights
CONFIDENCE_WEIGHTS = {
    'Technology': {
        'prediction': 0.65,  # Weight for GraphSAGE prediction
        'specificity': 0.2,  # Weight for entity specificity
        'context': 0.15,     # Weight for context quality
    },
    'Skill': {
        'prediction': 0.4,
        'specificity': 0.3,
        'context': 0.3,
    },
    'Soft_Skill': {
        'prediction': 0.4,
        'specificity': 0.3,
        'context': 0.3,
    },
    'Experience_Level': {
        'prediction': 0.5,
        'specificity': 0.25,
        'context': 0.25,
    },
    'default': {
        'prediction': 0.6,
        'specificity': 0.25,
        'context': 0.15,
    }
}