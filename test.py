import numpy as np


# Define a function to compute Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


# Function to ensure diversity based on Euclidean distance
def ensure_diversity(vectors, threshold):
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):  # Compare each pair once
            dist = euclidean_distance(vectors[i], vectors[j])
            if dist < threshold:
                print(f"Vectors {i} and {j} are too similar! Distance: {dist}")
                # Perform some action, e.g., mutate or replace one of the vectors
                # vectors[j] = mutate(vectors[j])


# Example list of weight vectors
population = [[0.1, 0.2, 0.3], [0.15, 0.22, 0.32], [0.9, 0.85, 0.88], [0.11, 0.21, 0.29]]
population = [[0.1, 0.2, 0.3]] * 3

# Define your threshold for "similarity"
similarity_threshold = 0.1

# Ensure diversity in the population
ensure_diversity(population, similarity_threshold)
