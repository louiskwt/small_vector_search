import heapq
import json
import os
from typing import List, Dict, Any
from utils import euclidean_distance, consine_similarity

class VectorStore():
    """A a small, self-contained Python class that acts as a tiny local vector database."""


    def __init__(self, filename="output.json"):
        self.filename = filename
        self.vectors: List[Dict[str, Any]] = []
        self._load()


    def add(self, vectors: list, id=None):
        data = {
            "vectors": vectors,
            "meta_data": id
        }
        self.vectors.append(data)


    def _load(self) -> None:
        """Load vectors from the JSON file if it exists."""
        if not os.path.exists(self.filename):
            return
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.vectors = data
        except (json.JSONDecodeError, OSError):
            print("Error when loading file")
            self.vectors = []

    def save(self) -> None:
        """Save the current vectors to the JSON file."""
        directory = os.path.dirname(self.filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp_filename = self.filename + ".tmp"

        with open(tmp_filename, "w", encoding="utf-8") as f:
            json.dump(self.vectors, f, indent=2)

        os.replace(tmp_filename, self.filename)


    def query(self, vectors, metric="euclidean", top_k=3):
        if metric == "euclidean":
            distances = [(euclidean_distance(d["vectors"], vectors), index) for index, d in enumerate(self.vectors)]
            qualified_distances = heapq.nsmallest(top_k, distances, key=lambda d: d[0])
            return [{"meta_data": self.vectors[index]["meta_data"], "vectors": self.vectors[index]["vectors"], "score": score} for score, index in qualified_distances]
        if metric == "cosine":
            similarities =  [(consine_similarity(d["vectors"], vectors), index) for index, d in enumerate(self.vectors)]
            qualified_similarites = heapq.nlargest(top_k, similarities, key=lambda s: s[0])
            return [{"meta_data": self.vectors[index]["meta_data"], "vectors": self.vectors[index]["vectors"], "score": score} for score, index in qualified_similarites]
        else:
            raise Exception("Cannot operate on unsupported metric.")
    
        


