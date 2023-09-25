from fastapi import FastAPI
import uvicorn
from typing import Union
import faiss
import numpy as np

app = FastAPI()
dims = 10
index = None


def parse_string(vec: str) -> list[float]:
    l = vec.split(',')
    if len(l) != dims:
        return None
    return [float(el) for el in l]

@app.on_event("startup")
def start():
    global index
    n_cells = 500

    quantizer = faiss.IndexFlatL2(dims)
    index = faiss.IndexIVFFlat(quantizer, dims, n_cells)

    arr = np.ascontiguousarray(np.random.random([5000, dims]))
    index.train(arr)
    index.add(arr)

@app.get("/")
def main() -> dict:
    return{'status': 'ok', 'message': 'Hello World'}

@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global index
    if item is None:
        return {'status': 'error', 'message': 'No item provided'}
    
    vec = parse_string(item)
    vec = np.ascontiguousarray(vec, dtype='float')[np.newaxis, :]

    knn, idx = index.search(vec, k=5)

    return {'status': 'ok', 'data': [str(el) for el in idx]}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)