from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.load_docs import DocChunk

class EmbedIndex:

    def __init__(self,model:str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model= SentenceTransformer(model)
        self.chunks=[]
        self.index= None


    def build(self,chunks):

        self.chunks=chunks
        #Extracting text from each chunk
        #print("DocChunk:",chunks)
        texts= [chunk.text for chunk in chunks]
        # texts into vectors
        embeddings = self.model.encode(texts,show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)


        # FAISS : calculate inner product between vectors and finds  top-k chunks with highest similarity to the query vector

        #print("Emedding Shape:",embeddings.shape)
        dim = embeddings.shape[1]
        #print(dim)
        self.index = faiss.IndexFlatIP(dim)  # flat = allvector, IP = inner product (cosine)
        self.index.add(embeddings)


    def retrieve(self, query:str, top_k:int =5):

        # Embedding the query
        query_embed= self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True)
        #Faiss to find top similarity 
        similarity_score, top_k_indices= self.index.search(query_embed,top_k)

        # Retrieving the actual vectors 
        top_k_vectors = []
        for i in top_k_indices[0]:
            # -1 represent missing 
            if i != -1:
                top_k_vectors.append(self.chunks[i])

        return top_k_vectors
                
    