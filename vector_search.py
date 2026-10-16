import math
import heapq

class VectorStore():
    """A a small, self-contained Python class that acts as a tiny local vector database."""

    filename = None

    def __init__(self, filename="output.json"):
        self.vector_data = []
        self.filename = filename
        

    def _add(self, vectors: list, id=None):
        data = {
            "vectors": vectors,
            "meta_data": id
        }
        self.vector_data.append(data)


    def euclidean_distance(v1: list, v2: list):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


    def consine_similarity(a: list, b: list):
        dot_product = sum([x * y for x, y in zip(a, b)])
        mag_a = math.sqrt(sum([x * x for x in a]))
        mag_b = math.sqrt(sum([x * x for x in b]))
        return dot_product / (mag_a * mag_b)       

    
    def query(self, vectors, metric="euclidean", top_k=3):
        if metric == "euclidean":
            distances = [(self.euclidean_distance(d.vectors, vectors), index) for index, d in enumerate(self.vector_data)]
            qualified_distances = heapq.nsmallest(top_k, distances, key=lambda d: d[0])
            return [self.vector_data[qd[1]] for qd in qualified_distances]
        if metric == "cosine":
            similarities =  [(self.consine_similarity(d.vectors, vectors), index) for index, d in enumerate(self.vector_data)]
            qualified_similarites = heapq.nlargest(top_k, similarities, key=lambda s: s[0])
            return [self.vector_data[qs[1]] for qs in qualified_similarites]
        else:
            raise Exception("Cannot operate on unsupported metric.")
    
        


