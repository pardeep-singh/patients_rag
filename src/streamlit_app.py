"""
Streamlit UI for Patient RAG System

A web interface for searching patient data using natural language queries.
"""

import streamlit as st
import os
from pathlib import Path
from search_patients import search_patients, format_results, PatientRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Page configuration
st.set_page_config(
    page_title="Patient RAG Search",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .metadata-badge {
        display: inline-block;
        background-color: #e3f2fd;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .similarity-score {
        color: #4caf50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_store(persist_dir="db", collection_name="patients_collection"):
    """Load the vector store (cached for performance)."""
    if not Path(persist_dir).exists():
        return None
    
    if not os.getenv('OPENAI_API_KEY'):
        return None
    
    try:
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embedder
        )
        return vectordb
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


def format_result_card(doc, index):
    """Format a single result as a card."""
    metadata = doc.metadata
    
    # Build metadata badges
    badges = []
    if 'age' in metadata:
        badges.append(f"👤 Age: {metadata['age']} years")
    if 'gender' in metadata:
        gender_full = "Male" if metadata['gender'] == 'M' else "Female"
        badges.append(f"⚧️ Gender: {gender_full}")
    if 'similarity_score' in metadata:
        score = metadata['similarity_score']
        badges.append(f"📊 Similarity: {score:.3f}")
    if 'patient_uid' in metadata:
        badges.append(f"🆔 UID: {metadata['patient_uid']}")
    
    # Create card HTML
    card_html = f"""
    <div class="result-card">
        <h3>Result #{index + 1}</h3>
        <div style="margin-bottom: 1rem;">
            {' '.join([f'<span class="metadata-badge">{badge}</span>' for badge in badges])}
        </div>
        <div style="margin-top: 1rem;">
            <strong>Patient Description:</strong>
            <p style="text-align: justify; line-height: 1.6;">{doc.page_content}</p>
        </div>
    </div>
    """
    return card_html


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🏥 Patient RAG Search System</div>', unsafe_allow_html=True)
    st.markdown("Search patient case reports using natural language queries with metadata filtering.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check API key
        if not os.getenv('OPENAI_API_KEY'):
            st.error("⚠️ OPENAI_API_KEY not set")
            st.info("Please set your OpenAI API key as an environment variable to use this app.")
            st.stop()
        else:
            st.success("✅ API Key configured")
        
        # Database settings
        st.subheader("Database Settings")
        persist_dir = st.text_input("Database Directory", value="db")
        collection_name = st.text_input("Collection Name", value="patients_collection")
        
        # Search settings
        st.subheader("Search Settings")
        k = st.slider("Number of Results", min_value=1, max_value=20, value=10)
        score_threshold = st.slider("Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        
        # Load vector store
        with st.spinner("Loading vector database..."):
            vectordb = load_vector_store(persist_dir, collection_name)
        
        if vectordb is None:
            st.error("❌ Could not load vector database")
            st.info("""
            Please ensure:
            1. The database directory exists
            2. You have run `python index_patients.py` first
            3. The collection name matches
            """)
            st.stop()
        else:
            try:
                doc_count = vectordb._collection.count()
                st.success(f"✅ Database loaded: {doc_count} documents")
            except:
                st.success("✅ Database loaded")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Give me male patients over 60 years that have been diagnosed with diabetes",
            "female patients with COVID-19",
            "patients over 50 years with heart disease",
            "male patients under 40 with diabetes",
            "female patients over 70 years",
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.query_input = example
                st.rerun()
    
    # Main content area
    st.markdown("---")
    
    # Query input
    # Initialize query_input in session state if not present
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ''
    
    query = st.text_input(
        "🔍 Enter your search query",
        value=st.session_state.query_input,
        placeholder="e.g., 'Give me male patients over 60 years that have been diagnosed with diabetes'",
        key="query_input"
    )
    
    # Advanced filters (optional)
    with st.expander("🔧 Advanced Filters (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            gender_filter = st.selectbox(
                "Gender",
                options=["Any", "Male", "Female"],
                index=0
            )
        with col2:
            age_min = st.number_input("Minimum Age", min_value=0, max_value=120, value=0)
            age_max = st.number_input("Maximum Age", min_value=0, max_value=120, value=120)
    
    # Search button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.query_input = ""
        st.rerun()
    
    # Perform search
    if search_button and query:
        with st.spinner("Searching patient database..."):
            try:
                # Build explicit filters if provided
                explicit_filters = None
                if gender_filter != "Any" or age_min > 0 or age_max < 120:
                    explicit_filters = {}
                    if gender_filter == "Male":
                        explicit_filters['gender'] = 'M'
                    elif gender_filter == "Female":
                        explicit_filters['gender'] = 'F'
                    if age_min > 0:
                        explicit_filters['age_min'] = age_min
                    if age_max < 120:
                        explicit_filters['age_max'] = age_max
                
                # Perform search
                results = search_patients(
                    query=query,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    k=k,
                    score_threshold=score_threshold
                )
                
                # Apply explicit filters if provided (post-filtering)
                if explicit_filters:
                    filtered_results = []
                    for doc in results:
                        metadata = doc.metadata
                        match = True
                        
                        if 'gender' in explicit_filters:
                            if metadata.get('gender') != explicit_filters['gender']:
                                match = False
                        if 'age_min' in explicit_filters:
                            if metadata.get('age', 0) < explicit_filters['age_min']:
                                match = False
                        if 'age_max' in explicit_filters:
                            if metadata.get('age', 200) > explicit_filters['age_max']:
                                match = False
                        
                        if match:
                            filtered_results.append(doc)
                    
                    results = filtered_results[:k]
                
                # Display results
                if results:
                    st.markdown(f"### 📊 Found {len(results)} matching patients")
                    st.markdown("---")
                    
                    # Display results
                    for i, doc in enumerate(results):
                        st.markdown(format_result_card(doc, i), unsafe_allow_html=True)
                    
                    # Download results option
                    st.markdown("---")
                    results_text = format_results(results, max_length=10000)
                    st.download_button(
                        label="📥 Download Results as Text",
                        data=results_text,
                        file_name="patient_search_results.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No results found. Try adjusting your query or filters.")
                    
            except Exception as e:
                st.error(f"Error during search: {e}")
                st.exception(e)
    
    elif search_button and not query:
        st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Patient RAG Search System | Built with Streamlit & LangChain</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

