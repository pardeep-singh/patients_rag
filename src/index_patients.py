"""
Patient RAG Indexing Script

This script indexes patient data from PMC-Patients.csv into a ChromaDB vector store.
It processes the top 1000 rows and creates searchable documents with metadata.
"""

import pandas as pd
import ast
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os


def parse_age(age_str):
    """Parse age from string format like '[[60.0, 'year']]' to float."""
    try:
        # Handle string representation of list
        if isinstance(age_str, str):
            # Try to evaluate as Python literal
            age_list = ast.literal_eval(age_str)
            if isinstance(age_list, list) and len(age_list) > 0:
                if isinstance(age_list[0], list) and len(age_list[0]) > 0:
                    return float(age_list[0][0])
                return float(age_list[0])
        elif isinstance(age_str, list):
            if len(age_str) > 0:
                if isinstance(age_str[0], list) and len(age_str[0]) > 0:
                    return float(age_str[0][0])
                return float(age_str[0])
        return None
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Warning: Could not parse age '{age_str}': {e}")
        return None


def create_documents(df, max_rows=1000):
    """
    Create LangChain documents from patient dataframe.
    
    Args:
        df: DataFrame with patient data
        max_rows: Maximum number of rows to process
        
    Returns:
        List of Document objects
    """
    documents = []
    df_subset = df.head(max_rows)
    
    for idx, row in df_subset.iterrows():
        # Get patient text
        patient_text = str(row['patient']) if pd.notna(row['patient']) else ""
        
        if not patient_text.strip():
            continue
        
        # Parse age
        age = parse_age(row['age'])
        
        # Get gender
        gender = str(row['gender']).strip().upper() if pd.notna(row['gender']) else None
        
        # Create metadata
        metadata = {
            'patient_uid': str(row.get('patient_uid', idx)),
            'patient_id': int(row.get('patient_id', idx)),
            'PMID': str(row.get('PMID', '')),
            'title': str(row.get('title', ''))[:200] if pd.notna(row.get('title')) else '',
        }
        
        # Add age and gender to metadata (for filtering)
        if age is not None:
            metadata['age'] = age
        if gender:
            metadata['gender'] = gender
        
        # Create document with patient text as content
        doc = Document(
            page_content=patient_text,
            metadata=metadata
        )
        
        documents.append(doc)
    
    return documents


def index_patients(
    csv_path="PMC-Patients.csv",
    persist_dir="db",
    collection_name="patients_collection",
    max_rows=1000,
    embedding_model="text-embedding-3-small"
):
    """
    Index patient data into ChromaDB vector store.
    
    Args:
        csv_path: Path to the CSV file
        persist_dir: Directory to persist the vector database
        collection_name: Name of the ChromaDB collection
        max_rows: Maximum number of rows to index
        embedding_model: OpenAI embedding model to use
    """
    print(f"Loading patient data from {csv_path}...")
    
    # Check if CSV exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Create documents
    print("Creating documents...")
    documents = create_documents(df, max_rows=max_rows)
    print(f"Created {len(documents)} documents")
    
    # Initialize embeddings
    print(f"Initializing embeddings with model: {embedding_model}...")
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    embedder = OpenAIEmbeddings(model=embedding_model)
    
    # Remove existing database if it exists
    if Path(persist_dir).exists():
        print(f"Removing existing database at {persist_dir}...")
        import shutil
        shutil.rmtree(persist_dir)
    
    # Create vector database
    print("Creating vector database...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector database created successfully!")
    print(f"Total documents indexed: {vectordb._collection.count()}")
    print(f"Database persisted to: {persist_dir}")
    
    return vectordb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index patient data into ChromaDB")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="PMC-Patients.csv",
        help="Path to the CSV file"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum number of rows to index"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="db",
        help="Directory to persist the vector database"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="patients_collection",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use"
    )
    
    args = parser.parse_args()
    
    try:
        vectordb = index_patients(
            csv_path=args.csv_path,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            max_rows=args.max_rows,
            embedding_model=args.embedding_model
        )
        print("\n✅ Indexing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        raise

