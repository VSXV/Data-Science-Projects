import numpy as np
import pandas as pd
import requests

def main():
    # Загрузка данных из файла CSV
    df_train = pd.read_csv(
        "/Users/vs/Programming/Data Science/Data-Science-Projects/12. grocery matching/data/train.csv", 
        index_col=0
    )
    
    # Удаление столбца 'Target' из данных
    df_train.drop("Target", axis=1, inplace=True)

    # Удаление указанных столбцов из данных
    for column in ['6', '21', '25', '33', '44', '59', '65', '70']:
        df_train.drop(column, axis=1, inplace=True)

    # Нормализация данных
    df_train = df_train / np.linalg.norm(df_train, axis=1)[:, np.newaxis]

    # Преобразование данных в формат numpy array и затем в строку
    item_data = np.ascontiguousarray(df_train.values).astype('float32')
    item_str = ','.join([str(el) for el in item_data[0]])  # Преобразование вектора в строку

    # Отправка GET-запроса к FastAPI-приложению для поиска ближайших соседей
    r = requests.get(f'http://localhost:8031/knn?item={item_str}')

    if r.status_code == 200:
        print(r.text)  # Вывод результатов поиска
    else:
        print('Error')  # Вывод сообщения об ошибке

if __name__ == "__main__":
    main()
