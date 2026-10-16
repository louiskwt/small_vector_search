from vector_search import VectorStore

# Simple 3-dimensional vectors
sample_data = [
    {"id": "cat",   "vector": [0.9, 0.1, 0.2]},
    {"id": "dog",   "vector": [0.8, 0.2, 0.1]},
    {"id": "car",   "vector": [0.1, 0.9, 0.3]},
    {"id": "bike",  "vector": [0.2, 0.8, 0.4]},
    {"id": "apple", "vector": [0.3, 0.2, 0.9]},
]

# A query that should be closest to "cat" / "dog"
query_animal = [0.85, 0.15, 0.15]

# A query that should be closest to "car" / "bike"
query_vehicle = [0.15, 0.85, 0.35]


store = VectorStore("demo.json")

store.add([0.9, 0.1, 0.2], id="cat")
store.add([0.8, 0.2, 0.1], id="dog")
store.add([0.1, 0.9, 0.3], id="car")
store.add([0.2, 0.8, 0.4], id="bike")
store.save()

print("Cosine – animal query:")
for r in store.query(query_animal, metric="cosine", top_k=3):
    print(f"  {r['meta_data']}: {r['vectors']} | Score: {r['score']:.4f}")

print("\nEuclidean – animal query:")
for r in store.query(query_animal, metric="euclidean", top_k=3):
    print(f"  {r['meta_data']}: {r['vectors']} | Score: {r['score']:.4f}")