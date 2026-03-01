# SHL Assessment Recommender

A semantic search engine that recommends SHL assessments based on job descriptions or hiring queries.  
Built with **Python**, **Sentence Transformers**, **FAISS**, and **Streamlit**.

## Features

- Accepts natural language queries (e.g., “I need a 40‑minute Java test for developers”).
- Retrieves the most relevant SHL assessment URLs by matching against a training set of query–URL pairs.
- Fast similarity search using FAISS.
- Simple web interface built with Streamlit.

## Dataset

The model is trained on the `Train-Set` sheet of `shl_catalog.xlsx`, which contains:
- `Query`: job descriptions or hiring requests.
- `Assessment_url`: corresponding SHL product page URLs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/shl-recommender.git
   cd shl-recommender
