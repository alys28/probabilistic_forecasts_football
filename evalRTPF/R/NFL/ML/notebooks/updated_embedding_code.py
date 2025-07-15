# Embed the training data
import faiss
import numpy as np
import pickle

# Dictionary to store metadata for each timestep
all_metadata = {}

for timesteps in models.keys():
    print(f"Embedding data for timesteps {timesteps}")
    timesteps_split = timesteps.split("-")
    start_time = float(timesteps_split[0])
    end_time = float(timesteps_split[1])
    relevant_data = []
    
    for timestep in training_data.keys():
        if timestep >= start_time and timestep < end_time or (end_time == 1.0 and timestep == 1.0):
            for row in training_data[timestep]:
                relevant_data.append(row)
    
    model = models[timesteps]
    
    # Embed the test data - get numpy arrays for FAISS
    embeddings, labels, metadata = model.embed_data(relevant_data, return_numpy=True)
    
    # Store metadata for this timestep
    all_metadata[timesteps] = metadata
    
    # Create binary labels from metadata (already computed in embed_data)
    binary_labels = [meta["binary_label"] for meta in metadata]
    
    # Save metadata to file
    with open(f"saved_indexes/metadata_{timesteps}.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Data shape: {embeddings.shape}")
    print(f"Number of positive labels: {sum(binary_labels)}")
    print(f"Number of negative labels: {len(binary_labels) - sum(binary_labels)}")
    
    # Create FAISS index for each timesteps
    # Make sure embeddings are float32 for FAISS
    embeddings = embeddings.astype(np.float32)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use shape[1] for embedding dimension
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, f"saved_indexes/faiss_index_{timesteps}.faiss")
    
    # Save labels separately
    np.save(f"saved_indexes/labels_{timesteps}.npy", np.array(labels))
    np.save(f"saved_indexes/binary_labels_{timesteps}.npy", np.array(binary_labels))
    
    print(f"FAISS index for timesteps {timesteps} created with {embeddings.shape[0]} vectors")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print()

print("All FAISS indexes created successfully!")

# Example: How to load and use the saved data later
def load_faiss_data(timesteps):
    """Helper function to load FAISS index and associated data"""
    # Load FAISS index
    index = faiss.read_index(f"saved_indexes/faiss_index_{timesteps}.faiss")
    
    # Load labels
    labels = np.load(f"saved_indexes/labels_{timesteps}.npy")
    binary_labels = np.load(f"saved_indexes/binary_labels_{timesteps}.npy")
    
    # Load metadata
    with open(f"saved_indexes/metadata_{timesteps}.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return index, labels, binary_labels, metadata

# Example usage of the helper function:
# index, labels, binary_labels, metadata = load_faiss_data("0.0-0.25")
# 
# # Now you can search for similar vectors
# query_vector = np.random.rand(1, embeddings.shape[1]).astype(np.float32)
# distances, indices = index.search(query_vector, k=5)  # Find 5 nearest neighbors
# 
# # Get metadata for the found neighbors
# neighbor_metadata = [metadata[i] for i in indices[0]] 