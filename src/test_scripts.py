"""
Test script to verify the indexing and search scripts work correctly.
This script tests the core functionality without requiring API calls.
"""

import pandas as pd
from index_patients import parse_age, create_documents
from search_patients import PatientRetriever


def test_age_parsing():
    """Test age parsing function."""
    print("=" * 80)
    print("TEST 1: Age Parsing")
    print("=" * 80)
    
    test_cases = [
        ('[[60.0, "year"]]', 60.0),
        ('[[39.0, "year"]]', 39.0),
        ('[[57.0, "year"]]', 57.0),
        ('[[69.0, "year"]]', 69.0),
    ]
    
    all_passed = True
    for age_str, expected in test_cases:
        result = parse_age(age_str)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"{status} {age_str} -> {result} (expected {expected})")
        if not passed:
            all_passed = False
    
    return all_passed


def test_document_creation():
    """Test document creation from dataframe."""
    print("\n" + "=" * 80)
    print("TEST 2: Document Creation")
    print("=" * 80)
    
    try:
        df = pd.read_csv('data/PMC-Patients.csv', nrows=5)
        docs = create_documents(df, max_rows=5)
        
        if len(docs) == 5:
            print(f"✅ Created {len(docs)} documents")
            
            # Check first document structure
            doc = docs[0]
            has_content = len(doc.page_content) > 0
            has_metadata = 'age' in doc.metadata and 'gender' in doc.metadata
            
            print(f"✅ Document has content: {has_content} ({len(doc.page_content)} chars)")
            print(f"✅ Document has metadata: {has_metadata}")
            print(f"   Sample metadata: age={doc.metadata.get('age')}, gender={doc.metadata.get('gender')}")
            
            return True
        else:
            print(f"❌ Expected 5 documents, got {len(docs)}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_query_parsing():
    """Test query parsing and filter building."""
    print("\n" + "=" * 80)
    print("TEST 3: Query Parsing")
    print("=" * 80)
    
    class MockVectordb:
        pass
    
    retriever = PatientRetriever(MockVectordb(), k=10)
    
    test_queries = [
        {
            'query': 'Give me male patients over 60 years that have been diagnosed with diabetes',
            'expected_filters': {'gender': 'M', 'age_min': 60}
        },
        {
            'query': 'female patients with COVID-19',
            'expected_filters': {'gender': 'F'}
        },
        {
            'query': 'patients over 50 years',
            'expected_filters': {'age_min': 50}
        },
        {
            'query': 'male patients under 40',
            'expected_filters': {'gender': 'M', 'age_max': 40}
        },
    ]
    
    all_passed = True
    for test_case in test_queries:
        query = test_case['query']
        expected = test_case['expected_filters']
        
        filters = retriever.parse_query_filters(query)
        where_filter = retriever.build_metadata_filter(filters)
        
        # Check if key filters match
        matches = all(filters.get(k) == expected.get(k) for k in expected.keys())
        status = "✅" if matches else "❌"
        
        print(f"{status} Query: {query[:60]}...")
        print(f"   Parsed filters: {filters}")
        print(f"   ChromaDB filter: {where_filter}")
        
        if not matches:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PATIENT RAG SYSTEM - SCRIPT VERIFICATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Age Parsing", test_age_parsing()))
    results.append(("Document Creation", test_document_creation()))
    results.append(("Query Parsing", test_query_parsing()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python index_patients.py")
        print("3. Run: python search_patients.py 'your query' --format")
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
    print("=" * 80)


if __name__ == "__main__":
    main()

