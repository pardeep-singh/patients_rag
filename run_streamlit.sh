#!/bin/bash
# Script to run the Streamlit app

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   The app will not work without it."
    echo ""
fi

# Run Streamlit
echo "🚀 Starting Streamlit app..."
echo "   Open your browser to the URL shown below"
echo ""
uv run streamlit run src/streamlit_app.py

