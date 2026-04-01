import torch

def similarity(a, b):
    return torch.dot(a, b)

def retrieve(query_vec, doc_vecs, docs):
    scores = [similarity(query_vec, d) for d in doc_vecs]
    idx = torch.argmax(torch.tensor(scores))
    return docs[idx]