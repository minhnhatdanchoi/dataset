# Knowledge Graph Entity Extraction

This project uses graph neural networks and natural language processing to extract entities and relationships from unstructured text in employee profiles, building a comprehensive knowledge graph in Neo4j.

## Project Structure

The code has been reorganized into a modular structure for better maintainability:

```
knowledge-graph-extraction/
├── main.py                # Main execution script
├── config.py              # Configuration parameters
├── entity_patterns.py     # Regex patterns for entity extraction
├── context_keywords.py    # Keywords for context evaluation
├── utils.py               # Utility functions
└── README.md              # Project documentation
```

## Features

- **Graph-based Entity Extraction**: Uses GraphSAGE to learn node embeddings and predict relationships
- **Confidence Scoring**: Evaluates extracted entities based on prediction score, specificity, and context
- **Automatic Relationship Detection**: Determines appropriate relationship types based on entity types
- **Knowledge Graph Integration**: Updates Neo4j graph database with discovered entities and relationships

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- py2neo
- Neo4j database

## Usage

1. Set up your Neo4j database and update the connection parameters in `config.py`
2. Run the main script:

```bash
python main.py
```

## How It Works

1. **Data Extraction**: The system connects to Neo4j and extracts existing nodes and relationships
2. **Graph Construction**: Constructs a graph structure from extracted data for model training
3. **Model Training**: Trains a GraphSAGE-based link prediction model
4. **Entity Extraction**: Processes "About" text from employee profiles to extract entities
5. **Relationship Prediction**: Uses the trained model to predict relationships between employees and extracted entities
6. **Knowledge Graph Update**: Updates the Neo4j database with new entities and relationships

## Customization

You can customize the system by modifying:

- **Entity patterns** in `entity_patterns.py` to target specific information
- **Context keywords** in `context_keywords.py` to improve context evaluation
- **Confidence thresholds** in `config.py` to adjust sensitivity for different entity types
- **Model parameters** in `config.py` to tune performance

## Output

The system provides detailed console output showing:
- Extracted entities from each employee profile
- Confidence scores and prediction details
- New relationships added to the knowledge graph
- Summary statistics of discovered nodes and relationships