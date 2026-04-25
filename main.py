from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample document
document = """
Machine learning is a field of artificial intelligence.
It focuses on training models to learn patterns from data.
Supervised learning uses labeled data.
Unsupervised learning finds hidden patterns.
"""

# Split document into sentences
sentences = document.split(".")

# Vectorization
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sentences)

def get_answer(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors)
    index = similarity.argmax()
    return sentences[index]

# User interaction
while True:
    query = input("Ask something (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = get_answer(query)
    print("Answer:", answer)