# Step 1: Set up the Knowledge Base and Implement Search Tools
# We will begin with a small corpus of a dataset as an example, and show how we can build a search method that finds the most related document given an input query. 
# We will use this corpus as the database and the method as the action space for building a GPT-based agent.


# First, let's import necessary packages

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
import pandas as pd
from collections import Counter, defaultdict\


# load dataset
recipes_data = pd.read_csv("data/test_recipes.csv")
print(recipes_data.head(10))

# clean dataset

# for each row in dataset, convert into corpus entry


# A toy corpus mimicing the documents of Wikipedia
CORPUS = [
    {
        "id": "doc1",
        "title": "Vincent van Gogh",
        "text": (
            "Vincent van Gogh was a Dutch post-impressionist painter who is among the most famous and influential figures "
            "in the history of Western art. He created about 2,100 artworks, including The Starry Night, while staying at "
            "the Saint-Paul-de-Mausole asylum in Saint-Rémy-de-Provence in 1889."
        ),
    },
    {
        "id": "doc2",
        "title": "The Starry Night",
        "text": (
            "The Starry Night is an oil-on-canvas painting by Dutch Post-Impressionist painter Vincent van Gogh. "
            "Painted in June 1889, it depicts the view from the east-facing window of his asylum room at Saint-Rémy-de-Provence, "
            "just before sunrise, with the addition of an ideal village."
        ),
    },
    {
        "id": "doc3",
        "title": "Saint-Rémy-de-Provence",
        "text": (
            "Saint-Rémy-de-Provence is a commune in the Bouches-du-Rhône department in Southern France. The Saint-Paul-de-Mausole "
            "asylum is located here, where Vincent van Gogh stayed and painted several works including The Starry Night."
        ),
    },
    {
        "id": "doc4",
        "title": "Pythagorean theorem",
        "text": (
            "In mathematics, the Pythagorean theorem is a fundamental relation in Euclidean geometry among the three sides of a "
            "right-angled triangle: the square of the hypotenuse equals the sum of the squares of the other two sides."
        ),
    },
    {
        "id": "doc5",
        "title": "Claude Monet",
        "text": (
            "Claude Monet was a French painter, a founder of Impressionist painting. His works include the Water Lilies series, "
            "Haystacks, and Impression, Sunrise."
        ),
    },
    # A quick play around: Add some extra documents and watch how the GPT model explores, searches, and reasons through more scenarios in the final step.
]

# Then, we design a simple search method based on TF-IDF to retrieve information from the corpus.

# TF-IDF (Term Frequency–Inverse Document Frequency) is a method to find the most relevant passages for a query.

# 1. We will tokenize each document and the query into words.
# 2. We will compute TF (Term Frequency) to measure how often a word appears in a document. More frequent indicates more important within that document.
# 3. We will compute IDF (Inverse Document Frequency), which is used to downweight words that are common across many documents, like “the” or “and,” and upweight rarer words.
# 4. We will compute TF-IDF vectors (containing the TF-IDF score for each word) for both documents and the query, then compute cosine similarity between the query vector and each document vector.
# 5. We will compute cosine similarity between the query vector and each document vector.
# 6. We will implement a search method that finds the documents with the highest similarity scores as the top-k search results.
# 7. We note that this action space can mostly only retrieve a small part of a passage based on the exact passage name, which is weaker than state-of-the-art retrievers. The purpose is to simulate how the search method in Wikipedia and make models to retrieve via reasoning in language.

# As an extension of the project, you can redefine the search method in this code snippet to incorporate a more powerful search method.

# 1.  Tokenize the document into words
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

#     Get all the words of each document in the corpus
DOC_TOKENS = [tokenize(d["title"] + " " + d["text"]) for d in CORPUS]

#     Get all the words from the corpus
VOCAB = sorted(set(t for doc in DOC_TOKENS for t in doc))


# 2.  Compute term frequency (TF) for each doc
def compute_tf(tokens: List[str]) -> Dict[str, float]:
    # Input: A list of all the words in a document
    # Output: A dictionary of the frequency of each word

    # ===== TODO =====
    # implement the function to compute normalized term frequency: count of word / doc length
    counts = defaultdict(int)
    for token in tokens:
      counts[token] += 1
    length = max(1, len(tokens))
    
    return {token: counts[token] / length for token in counts}




# 3.   Compute the document frequency across corpus: how many docs does a word appear?
def compute_df(doc_tokens: List[List[str]]) -> Dict[str, float]:
    # Input: A list of lists of tokens in each document
    # Output: A dictionary of the counts of each word appearing across the documents

    # ===== TODO =====
    # implement the function to compute document frequency: count of the word appearing in the documents
    df = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    return df

#     Compute the inverse document frequency (higher for rarer terms), in which we use a smoothed variant
DF = compute_df(DOC_TOKENS) # Get the DF
N_DOC = len(DOC_TOKENS) # number of docs
IDF = {t: math.log((N_DOC + 1) / (DF[t] + 0.5)) + 1 for t in VOCAB} # Inverse document frequency



# 4.   We compute TF-IDF vectors for each document, which is the product between
def tfidf_vector(tokens: List[str]) -> Dict[str, float]:
    # Input: A list of words in a document
    # Output: A dictionary of tf-idf score of each word
    tf = compute_tf(tokens)
    vec = {t: tf[t] * IDF.get(t, 0.0) for t in tf}
    return vec

DOC_VECS = [tfidf_vector(tokens) for tokens in DOC_TOKENS]


# 5.   We compute the cosine similarity for the search
def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Inputs: Two dictrionaries of tf-idf vectors of two document
    # Output: The cosine similarity scalar between the two vector

    if not a or not b:
        return 0.0

    # ===== TODO =====
    # Compute the cosine similarity between two tf-idf vectors
    # Notice that they are two dictionaries and could have missing keys
    
    # compute dot product
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    
    # compute norms
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))

    # similarity
    return dot / (na * nb + 1e-12)


# 6.   We implement a search method based on the cosine similarity, which finds the documents with the highest similarity scores as the top-k search results.
def search_corpus(query: str, k: int = 3) -> List[Dict[str, Any]]:
    qvec = tfidf_vector(tokenize(query))
    scored = [(cosine(qvec, v), i) for i, v in enumerate(DOC_VECS)]
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:k]:
        d = CORPUS[idx].copy()
        d["score"] = float(score)
        results.append(d)
    return results  

#       Integrate the search method as a tool
def tool_search(query: str, k: int = 3) -> Dict[str, Any]:
    hits = search_corpus(query, k=k)
    # Return a concise, citation-friendly payload
    return {
        "tool": "search",
        "query": query,
        "results": [
            {"id": h["id"], "title": h["title"], "snippet": h["text"][:240] + ("..." if len(h["text"]) > 240 else "")}
            for h in hits
        ],
    }

TOOLS = {
    "search": {
        "schema": {"query": "str", "k": "int? (default=3)"},
        "fn": tool_search
    },
    "finish": {
        "schema": {"answer": "str"},
        "fn": lambda answer: {"tool": "finish", "answer": answer}
    }
}
