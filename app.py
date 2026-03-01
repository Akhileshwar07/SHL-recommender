import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# ============================================
# 1. LOAD THE EXCEL FILE (both sheets)
# ============================================
file_path = 'data/shl_catalog.xlsx'  # adjust if your file is elsewhere

# Load the training sheet
train_df = pd.read_excel(file_path, sheet_name='Train-Set')
# Load the test sheet (optional, if you want to see test queries)
test_df = pd.read_excel(file_path, sheet_name='Test-Set')

# ============================================
# 2. INSPECT COLUMNS (run once to verify)
# ============================================
# Uncomment the next lines to see column names in the terminal
# print("Train columns:", train_df.columns.tolist())
# print("Test columns:", test_df.columns.tolist())

# ============================================
# 3. PREPARE TRAINING DATA
# ============================================
# We'll use the 'Query' column as the text to search on,
# and 'Assessment_url' as the recommendation target.
# Note: Some queries have multiple URLs (multiple rows for same query).
# We'll keep each row as a separate item so that all URL associations are preserved.

train_texts = train_df['Query'].tolist()          # list of queries
train_urls = train_df['Assessment_url'].tolist()  # corresponding URLs

# ============================================
# 4. GENERATE EMBEDDINGS FOR ALL TRAINING QUERIES
# ============================================
model = SentenceTransformer('all-MiniLM-L6-v2')   # fast, good quality
print("Encoding training queries...")
train_embeddings = model.encode(train_texts, show_progress_bar=True)

# ============================================
# 5. BUILD FAISS INDEX FOR FAST SEARCH
# ============================================
embedding_matrix = np.array(train_embeddings).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance
index.add(embedding_matrix)

# ============================================
# 6. RECOMMENDATION FUNCTION
# ============================================
def recommend(query, top_k=5):
    """
    Given a query string, return the top_k most similar training queries
    and their associated URLs.
    """
    # Encode the user query
    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype('float32')

    # Search in FAISS
    distances, indices = index.search(query_emb, top_k)

    # Collect results (avoid duplicates if you want, but we'll show all)
    results = []
    for idx in indices[0]:
        results.append({
            'Matched Query': train_texts[idx],
            'Recommended URL': train_urls[idx]
        })
    return results

# ============================================
# 7. STREAMLIT USER INTERFACE
# ============================================
st.title("SHL Assessment Recommender")
st.markdown("Enter a hiring need or job description, and I'll recommend relevant SHL assessments.")

query = st.text_area("Your query:", height=150,
                     placeholder="e.g., I need a 40-minute Java test for developers...")

if st.button("Get Recommendations") and query:
    with st.spinner("Searching..."):
        results = recommend(query, top_k=10)   # you can change top_k

    st.success(f"Found {len(results)} recommendations:")
    for i, res in enumerate(results, 1):
        st.markdown(f"**{i}. {res['Recommended URL']}**")
        st.caption(f"Matched query: {res['Matched Query'][:200]}...")   # show snippet
        st.markdown("---")