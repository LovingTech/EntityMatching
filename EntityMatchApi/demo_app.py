import streamlit as st
import requests

# API endpoint
API_URL = "http://localhost:8000/search"  # Update if hosted elsewhere

st.title("Semantic Search 🔍 with Embeddings")

# User input
user_query = st.text_input("Enter a sentence to search:", "")

top_k = st.slider("Number of results", min_value=1, max_value=50, value=5)

if st.button("Search") and user_query.strip():
    with st.spinner("Getting matches..."):
        try:
            response = requests.post(API_URL, json={"text": user_query, "top_k": top_k})
            response.raise_for_status()
            results = response.json()["results"]

            st.success(f"Top {top_k} matches:")
            for i, match in enumerate(results, 1):
                st.markdown(f"**{i}.** {match['text']}")
                st.caption(f"Score: {match['score']:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
