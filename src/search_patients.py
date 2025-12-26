"""
Patient RAG Search Script

This script provides search functionality over the indexed patient database.
It supports both semantic search and metadata filtering for structured queries.
"""

import re
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import os


class PatientRetriever:
    """
    A retriever that combines semantic search with metadata filtering
    for patient queries.
    """
    
    def __init__(
        self,
        vectordb: Chroma,
        k: int = 10,
        score_threshold: float = 0.2
    ):
        """
        Initialize the patient retriever.
        
        Args:
            vectordb: ChromaDB vector store instance
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.vectordb = vectordb
        self.k = k
        self.score_threshold = score_threshold
    
    def parse_query_filters(self, query: str) -> Dict[str, Any]:
        """
        Parse structured filters from natural language query.
        
        Examples:
            "male patients over 60" -> {'gender': 'M', 'age_min': 60}
            "female patients" -> {'gender': 'F'}
            "patients over 50 years" -> {'age_min': 50}
        """
        filters = {}
        query_lower = query.lower()
        
        # Parse gender
        if re.search(r'\bmale\b', query_lower):
            filters['gender'] = 'M'
        elif re.search(r'\bfemale\b', query_lower):
            filters['gender'] = 'F'
        elif re.search(r'\bmen\b', query_lower):
            filters['gender'] = 'M'
        elif re.search(r'\bwomen\b', query_lower):
            filters['gender'] = 'F'
        
        # Parse age constraints
        # "over 60", "above 60", "more than 60", "60+"
        age_patterns = [
            (r'over\s+(\d+)', 'age_min'),
            (r'above\s+(\d+)', 'age_min'),
            (r'more\s+than\s+(\d+)', 'age_min'),
            (r'(\d+)\s*\+', 'age_min'),
            (r'older\s+than\s+(\d+)', 'age_min'),
            (r'under\s+(\d+)', 'age_max'),
            (r'below\s+(\d+)', 'age_max'),
            (r'less\s+than\s+(\d+)', 'age_max'),
            (r'younger\s+than\s+(\d+)', 'age_max'),
        ]
        
        for pattern, filter_key in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                age_value = int(match.group(1))
                if filter_key == 'age_min':
                    # If age_min already exists, take the maximum
                    filters['age_min'] = max(filters.get('age_min', 0), age_value)
                else:
                    # If age_max already exists, take the minimum
                    filters['age_max'] = min(filters.get('age_max', float('inf')), age_value)
                break
        
        return filters
    
    def build_metadata_filter(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """
        Build ChromaDB metadata filter from parsed filters.
        
        ChromaDB uses 'where' parameter with operators like $gte, $lte, $eq.
        For multiple conditions, we use $and operator.
        """
        if not filters:
            return None
        
        conditions = []
        
        # Gender filter
        if 'gender' in filters:
            conditions.append({'gender': {'$eq': filters['gender']}})
        
        # Age filters
        age_conditions = {}
        if 'age_min' in filters:
            age_conditions['$gte'] = filters['age_min']
        if 'age_max' in filters:
            age_conditions['$lte'] = filters['age_max']
        
        if age_conditions:
            conditions.append({'age': age_conditions})
        
        # Return single condition or $and for multiple
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {'$and': conditions}
        
        return None
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for patients matching the query.
        
        Args:
            query: Natural language search query
            k: Number of results to return (overrides default)
            filters: Optional explicit filters dict
            
        Returns:
            List of matching Document objects
        """
        # Use provided k or default
        search_k = k or self.k
        
        # Parse filters from query if not provided
        if filters is None:
            filters = self.parse_query_filters(query)
        
        # Build metadata filter for ChromaDB
        where_filter = self.build_metadata_filter(filters)
        
        # Perform semantic search with metadata filtering
        # LangChain Chroma uses 'filter' parameter (MongoDB-style)
        if where_filter:
            results = self.vectordb.similarity_search_with_score(
                query=query,
                k=search_k,
                filter=where_filter
            )
        else:
            # No metadata filters, just semantic search
            results = self.vectordb.similarity_search_with_score(
                query=query,
                k=search_k
            )
        
        # Filter by score threshold and return documents
        filtered_results = []
        for doc, score in results:
            # ChromaDB returns distance (lower is better), convert to similarity
            similarity = 1 - score if score <= 1 else 1 / (1 + score)
            if similarity >= self.score_threshold:
                # Add similarity score to metadata
                doc.metadata['similarity_score'] = similarity
                filtered_results.append(doc)
        
        return filtered_results


def format_results(documents: List[Document], max_length: int = 500) -> str:
    """
    Format search results for display.
    
    Args:
        documents: List of Document objects
        max_length: Maximum length of content preview
        
    Returns:
        Formatted string
    """
    if not documents:
        return "No results found."
    
    output = []
    output.append(f"Found {len(documents)} matching patients:\n")
    output.append("=" * 80)
    
    for i, doc in enumerate(documents, 1):
        output.append(f"\n[Result {i}]")
        output.append("-" * 80)
        
        # Display metadata
        metadata = doc.metadata
        if 'patient_uid' in metadata:
            output.append(f"Patient UID: {metadata['patient_uid']}")
        if 'age' in metadata:
            output.append(f"Age: {metadata['age']} years")
        if 'gender' in metadata:
            gender_full = "Male" if metadata['gender'] == 'M' else "Female"
            output.append(f"Gender: {gender_full}")
        if 'similarity_score' in metadata:
            output.append(f"Similarity Score: {metadata['similarity_score']:.3f}")
        
        output.append(f"\nPatient Description:")
        content = doc.page_content
        if len(content) > max_length:
            content = content[:max_length] + "..."
        output.append(content)
        output.append("")
    
    return "\n".join(output)


def search_patients(
    query: str,
    persist_dir: str = "db",
    collection_name: str = "patients_collection",
    k: int = 10,
    score_threshold: float = 0.2,
    embedding_model: str = "text-embedding-3-small"
):
    """
    Search for patients in the indexed database.
    
    Args:
        query: Search query string
        persist_dir: Directory where the vector database is persisted
        collection_name: Name of the ChromaDB collection
        k: Number of results to return
        score_threshold: Minimum similarity score
        embedding_model: OpenAI embedding model to use
        
    Returns:
        List of matching Document objects
    """
    # Check if database exists
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Database not found at {persist_dir}. "
            "Please run index_patients.py first."
        )
    
    # Initialize embeddings
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    embedder = OpenAIEmbeddings(model=embedding_model)
    
    # Load vector database
    vectordb = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embedder
    )
    
    # Create retriever and search
    retriever = PatientRetriever(
        vectordb=vectordb,
        k=k,
        score_threshold=score_threshold
    )
    
    results = retriever.search(query, k=k)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search patient database")
    parser.add_argument(
        "query",
        type=str,
        help="Search query (e.g., 'male patients over 60 years with diabetes')"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="db",
        help="Directory where the vector database is persisted"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="patients_collection",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.2,
        help="Minimum similarity score threshold"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use"
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="Format and display results nicely"
    )
    
    args = parser.parse_args()
    
    try:
        results = search_patients(
            query=args.query,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            k=args.k,
            score_threshold=args.score_threshold,
            embedding_model=args.embedding_model
        )
        
        if args.format:
            print(format_results(results))
        else:
            print(f"Found {len(results)} results")
            for i, doc in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Age: {doc.metadata.get('age', 'N/A')}")
                print(f"Gender: {doc.metadata.get('gender', 'N/A')}")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Similarity Score: {doc.metadata.get('similarity_score', 'N/A')}")
        
        print(f"\n✅ Search completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        raise

