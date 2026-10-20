<p align="center">
  <img src="https://raw.githubusercontent.com/MasterPandaa/MasterPandaa/main/assets/banners/JansenAI.svg" alt="JansenAI Dialogue Agent Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-NLP_Conversational_Agent-4F46E5?style=flat-square" alt="Domain">
  <img src="https://img.shields.io/badge/Library-NLTK_3.8+-0284C7?style=flat-square" alt="NLTK">
  <img src="https://img.shields.io/badge/Vector-TF--IDF_Similarity-10B981?style=flat-square" alt="TF-IDF">
  <img src="https://img.shields.io/badge/Language-Python_3-3776AB?style=flat-square&logo=python" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License"></a>
</p>

## Overview

JansenAI is a conversational dialogue agent built with Python, demonstrating core principles of statistical **Natural Language Processing (NLP)**. The engine transforms unstructured user queries into vector representations using tokenization, lemmatization, and TF-IDF weighting, matching intent against an indexed corpus using cosine similarity metrics.

---

## Core Capabilities

- **Linguistic Preprocessing Pipeline**: Tokenizes sentences, removes noise, strips punctuation, and applies WordNet lemmatization.
- **TF-IDF Vector Space Modeling**: Maps conversational query utterances and corpus documents into high-dimensional sparse vector spaces.
- **Cosine Proximity Retrieval**: Identifies the closest semantically aligned responses by computing vector angle dot products.
- **Fallback Intent Handling**: Employs configurable similarity score thresholds to detect out-of-domain queries and trigger clarification routines.
- **Interactive Terminal Console**: Lightweight execution interface requiring minimal runtime footprint and zero GPU accelerators.

---

## Quickstart

### Prerequisites

- Python 3.8+

### Setup and Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/MasterPandaa/JansenAI.git
   cd JansenAI
   ```

2. Install dependencies:
   ```bash
   pip install nltk scikit-learn
   ```

3. Download required NLTK corpora:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. Launch the agent:
   ```bash
   python chatbot.py
   ```

---

## License

This project is licensed under the [MIT License](LICENSE).
