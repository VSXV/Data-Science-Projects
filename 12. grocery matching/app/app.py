import faiss
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI

app = FastAPI()

# Инициализация переменных
dims = 1
index = None
k = 5

# Функция для разбора строки с вектором чисел
def parse_string(vec: str) -> list[float]:
    l = vec.split(',')
    if len(l) != dims:
        return None
    return [float(el) for el in l]

# Функция, выполняемая при старте приложения
@app.on_event("startup")
def start():
    # Загрузка данных и предварительная обработка
    df_base = pd.read_csv(
        '/Users/vs/Programming/Data Science/Data-Science-Projects/12. grocery matching/data/base.csv', 
        index_col=0
    )
    
    for column in ['6', '21', '25', '33', '44', '59', '65', '70']:
        df_base.drop(column, axis=1, inplace=True)

    df_base = df_base / np.linalg.norm(df_base, axis=1)[:, np.newaxis]

    global index
    global dims
    dims = df_base.shape[1]
    n_cells = 500

    # Инициализация Faiss Index для поиска ближайших соседей
    quantizer = faiss.IndexFlatL2(dims)
    index = faiss.IndexIVFFlat(quantizer, dims, n_cells)

    index.train(df_base)
    index.add(df_base)

# Основной маршрут, возвращающий сообщение о статусе приложения
@app.get("/")
def main() -> dict:
    return {'status': 'ok', 'message': 'Hello World'}

# Маршрут для поиска ближайших соседей
@app.get("/knn")
def match(item: str):
    global index
    global k
    if item is None:
        return {'status': 'error', 'message': 'No item provided'}
    
    # Преобразование строки в вектор и поиск ближайших соседей
    vec = parse_string(item)
    if vec is None:
        return {'status': 'error', 'message': 'Invalid item format'}

    vec_np = np.array(vec, dtype=np.float32)


    # Используйте vec_np для поиска ближайших соседей
    index.nprobe = 100
    knn, idx = index.search(vec_np.reshape(1, -1), k)

    return {'status': 'ok', 'data': [str(el) for el in idx[0]]}

if __name__ == "__main__":
    # Запуск приложения с использованием Uvicorn
    uvicorn.run(app, host="localhost", port=8031)