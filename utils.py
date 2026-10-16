import math

def euclidean_distance(v1: list, v2: list):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def consine_similarity(a: list, b: list):
    dot_product = sum([x * y for x, y in zip(a, b)])
    mag_a = math.sqrt(sum([x * x for x in a]))
    mag_b = math.sqrt(sum([x * x for x in b]))
    return dot_product / (mag_a * mag_b)       