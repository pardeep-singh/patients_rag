# Patients Record Retrieval System

A system for searching patient data from medical case records. This system indexes patient information and enables semantic search with metadata filtering.

## Features

- **Indexing**: Index patient data from CSV into a ChromaDB vector store
- **Semantic Search**: Find patients based on medical conditions and symptoms
- **Metadata Filtering**: Filter by age, gender, and other structured attributes
- **Hybrid Queries**: Combine natural language queries with structured filters

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### 1. Index Patient Data

Index the patient data from the CSV file into the vector database:

```bash
uv run src/index_patients.py --csv-path data/PMC-Patients.csv
```

Options:
- `--csv-path`: Path to CSV file (default: `PMC-Patients.csv`)
- `--max-rows`: Maximum number of rows to index (default: 1000)
- `--persist-dir`: Directory to persist database (default: `db`)
- `--collection-name`: ChromaDB collection name (default: `patients_collection`)
- `--embedding-model`: OpenAI embedding model (default: `text-embedding-3-small`)

Example:
```bash
uv run src/index_patients.py --csv-path data/PMC-Patients.csv
```

### 2. Search Patient Database

Search for patients using natural language queries:

```bash
uv run src/search_patients.py "your query here"
```

Options:
- `query`: The search query (required)
- `--persist-dir`: Database directory (default: `db`)
- `--collection-name`: Collection name (default: `patients_collection`)
- `--k`: Number of results (default: 10)
- `--score-threshold`: Minimum similarity score (default: 0.2)
- `--format`: Format and display results nicely

Examples:

```bash
# Search for male patients over 60 with diabetes
uv run src/search_patients.py "Give me male patients over 60 years that have been diagnosed with diabetes" --format

# Search for female patients
uv run src/search_patients.py "female patients with COVID-19" --format

# Search with age filter
uv run src/search_patients.py "patients over 50 years with heart disease" --k 5 --format
```

## Query Examples

The system understands natural language queries and automatically extracts:

- **Gender filters**: "male", "female", "men", "women"
- **Age filters**: "over 60", "above 50", "under 30", "60+"
- **Medical conditions**: Any condition mentioned in the query (via semantic search)

Examples:
- `"male patients over 60 years with diabetes"`
- `"female patients under 40 with COVID-19"`
- `"patients above 50 years diagnosed with heart disease"`
- `"younger than 30 years old patients"`

## Architecture

- **Vector Store**: ChromaDB with OpenAI embeddings
- **Embeddings**: OpenAI `text-embedding-3-small` model
- **Search**: Semantic similarity search with metadata filtering
- **Metadata**: Age (numeric), Gender (M/F), Patient UID, PMID, Title

## Data Structure

The system indexes the following fields:
- `patient`: Patient case description (used for semantic search)
- `age`: Patient age in years (numeric, for filtering)
- `gender`: Patient gender (M/F, for filtering)
- `patient_uid`: Unique patient identifier
- `PMID`: PubMed ID
- `title`: Article title

### 3. Streamlit Web UI (Optional)

Launch an interactive web interface for searching:

```bash
uv run streamlit run src/streamlit_app.py
```
or run the following script:
```bash
./run_streamlit.sh
```

## Query Examples

The system understands natural language queries and automatically extracts:

- **Gender filters**: "male", "female", "men", "women"
- **Age filters**: "over 60", "above 50", "under 30", "60+"
- **Medical conditions**: Any condition mentioned in the query (via semantic search)

Examples:
- `"male patients over 60 years with diabetes"`
- `"female patients under 40 with COVID-19"`
- `"patients above 50 years diagnosed with heart disease"`
- `"younger than 30 years old patients"`

## Architecture

- **Vector Store**: ChromaDB with OpenAI embeddings
- **Embeddings**: OpenAI `text-embedding-3-small` model
- **Search**: Semantic similarity search with metadata filtering
- **Metadata**: Age (numeric), Gender (M/F), Patient UID, PMID, Title
- **UI**: Streamlit web interface for interactive searching

## Data Structure

The system indexes the following fields:
- `patient`: Patient case description (used for semantic search)
- `age`: Patient age in years (numeric, for filtering)
- `gender`: Patient gender (M/F, for filtering)
- `patient_uid`: Unique patient identifier
- `PMID`: PubMed ID
- `title`: Article title
