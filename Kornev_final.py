import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

### Внимание!! Введите путь до файлов, указанных ниже
X_bp_raw = pd.read_excel("X_bp.xlsx").drop("Unnamed: 0", axis="columns")
X_nup_raw = pd.read_excel("X_nup.xlsx")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

## Соединяем два датасета в один при помощи INNER. Также удаляем Unnamed столбцы
## При первичной проверки датасета был замечен один выброс. Удаляем его сразу же.
X_all = X_bp_raw.merge(X_nup_raw, left_index=True, right_index=True,
                       how="inner").drop(columns=["Unnamed: 0"], axis = 1).drop([19], axis = 0)

## Проверим наличие пропусков

## Всё чисто, Null значений нет

## Теперь необходимо ознакомится с данными. Используем для этого гисторамму
## Данную функцию можно вызвать в любой момент для проверки данных, для этого необходимо изменять X_all


def Histogram_all():
    for i in range(len(X_all.columns)):
        plt.figure(figsize=(8,4))
        plt.ylabel("Элементы")
        plt.title("Модель Гистограммы: " + X_all.columns[i])
        sns.histplot(data = X_all.iloc[:, i])
        plt.show()
        print("Медианное значение: " + X_all.columns[i] + " равняется " + str(X_all.iloc[:, i].median()))
        print("Среднее значение: " + X_all.columns[i] + " равняется " + str(X_all.iloc[:, i].mean()))

## Ящик с усами
## Данные функции были сделаны отдельно друг от друга. Таким образом ненужные данные не будут мешать при анализе

def first_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Соотношение матрица-наполнитель"])
    plt.boxplot([X_all["Соотношение матрица-наполнитель"]], labels=[f"{number} компонента"])
    plt.title('Соотношение матрица-наполнитель', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def second_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Плотность, кг/м3"])
    plt.boxplot([X_all["Плотность, кг/м3"]], labels=[f"{number} компонента"])
    plt.title('Плотность, кг/м3', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def third_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["модуль упругости, ГПа"])
    plt.boxplot([X_all["модуль упругости, ГПа"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Модуль упругости, ГПа', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def fourth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Количество отвердителя, м.%"])
    plt.boxplot([X_all["Количество отвердителя, м.%"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Количество отвердителя, м.%', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def fifth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Содержание эпоксидных групп,%_2"])
    plt.boxplot([X_all["Содержание эпоксидных групп,%_2"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Содержание эпоксидных групп, %', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def sixth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Температура вспышки, С_2"])
    plt.boxplot([X_all["Температура вспышки, С_2"]], patch_artist=True, labels=[f"{number} компонента(ов)"])
    plt.title('Температура вспышки, С', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def seventh_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Поверхностная плотность, г/м2"])
    plt.boxplot([X_all["Поверхностная плотность, г/м2"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Поверхностная плотность, г/м2', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def eighth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Модуль упругости при растяжении, ГПа"])
    plt.boxplot([X_all["Модуль упругости при растяжении, ГПа"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Модуль упругости при растяжении, ГПа', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def ninth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Прочность при растяжении, МПа"])
    plt.boxplot([X_all["Прочность при растяжении, МПа"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Прочность при растяжении, МПа', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def tenth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Потребление смолы, г/м2"])
    plt.boxplot([X_all["Потребление смолы, г/м2"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Потребление смолы, г/м2', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def eleventh_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Угол нашивки, град"])
    plt.boxplot([X_all["Угол нашивки, град"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Угол нашивки, град', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def twelfth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Шаг нашивки"])
    plt.boxplot([X_all["Шаг нашивки"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Шаг нашивки', fontsize=16)
    plt.ylabel('Диапазон данных', fontsize=16)
    plt.show()


def thirteenth_column_boxplot():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    number = len(X_all["Угол нашивки, град"])
    plt.boxplot([X_all["Угол нашивки, град"]], patch_artist=True, labels=[f"{number} компонента"])
    plt.title('Плотность нашивки', fontsize=16)
    plt.ylabel('Плотность нашивки', fontsize=16)
    plt.show()


## Попарные графики рассеяния точек
def pairplot_graph():
    g = sns.pairplot(X_all, hue="Угол нашивки, град", height=2,
                     diag_kind='kde', kind='scatter', palette='cividis')
    g.set(ylabel=None)
    sns.set_style('ticks')
    g.tight_layout()
    plt.show()

## Тепловая карта
def heatmap():
    plt.figure(figsize=(18,10))
    sns.heatmap(X_all.corr(), cmap = "rainbow", annot = True)
    plt.show()

## При использовании тепловой карты чаще всего наблюдаем корреляцию, близкой к 0, из-за чего приходим к выводу, что
## в данных отсутствует линейная зависимость.

## Заменим значения параметра угла нашивки
le = LabelEncoder()
X_all["Угол нашивки, град"] = le.fit_transform(X_all["Угол нашивки, град"])


## Удаляем шумы при помощи Z-score
def check_outliers(X_bp_no_outliers):
    for i in range(len(X_bp_no_outliers.columns)):
        X_bp_no_outliers["Z-score"] = (X_bp_no_outliers.iloc[:, i] - X_bp_no_outliers.iloc[:, i].mean()) / X_bp_no_outliers.iloc[:, i].std()
        X_bp_no_outliers = X_bp_no_outliers[(X_bp_no_outliers["Z-score"]> - 3) & ((X_bp_no_outliers)["Z-score"]<3)]
    return X_bp_no_outliers.drop("Z-score", axis="columns")
X_all = check_outliers(X_all) ## Включение функции удаления шумов


## Нормализация данных
minmax_scaler = MinMaxScaler()
dataset_norm = minmax_scaler.fit_transform(np.array(X_all))
X_bp_no_MinMax = pd.DataFrame(data = dataset_norm, columns = X_all.columns)


def model_of_elasticity():
    X = X_bp_no_MinMax.drop(["Модуль упругости при растяжении, ГПа"], axis=1)
    y = X_bp_no_MinMax[["Модуль упругости при растяжении, ГПа"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.linear_model import RANSACRegressor

    model_params = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                "fit_intercept" : ["True", "False"]
            }
        },
        'RANSAC': {
            'model': RANSACRegressor(LinearRegression()),
            'params': {
                'max_trials': [1, 2, 4],
                "min_samples" : [1, 2],
                "loss" : ["absolute_error"],
                "residual_threshold": [1, 5, 10]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                "n_estimators" : [5, 10, 20, 100, 200],
                "max_depth" : [4, 5, 7, 10],
                "criterion" : ["squared_error"]
            }
        },
        "SVR" : {
           "model" : SVR(),
            "params": {
                "C" : [1, 3, 5, 10],
                "epsilon" : [0.2, 0.5, 1, 2],
                "kernel" : ["rbf"]
            }
        }
    }
    scores = []

    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
        clf.fit(X_train, y_train.values.ravel())
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })

    df = pd.DataFrame(scores, columns=['model', 'best_score', "best_params"])
    print(df)

    ## Возьмем за основу RandomForest
    model = RandomForestRegressor(max_depth=4, n_estimators=200)
    model.fit(X_train, y_train.values.ravel())
    print(model.predict(X_test))

    ## Теперь сравним предсказанные значения с реальными
    print(np.mean(np.abs(y_test.values.ravel()-model.predict(X_test))))


def model_of_durability():
    X = X_bp_no_MinMax.drop(["Прочность при растяжении, МПа"], axis=1)
    y = X_bp_no_MinMax[["Прочность при растяжении, МПа"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.linear_model import RANSACRegressor

    model_params = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                "fit_intercept": ["True", "False"]
            }
        },
        'RANSAC': {
            'model': RANSACRegressor(LinearRegression()),
            'params': {
                'max_trials': [1, 2, 4],
                "min_samples": [1, 2],
                "loss": ["absolute_error"],
                "residual_threshold": [1, 5, 10]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                "n_estimators": [5, 10, 20, 100, 200],
                "max_depth": [4, 5, 7, 10],
                "criterion": ["squared_error"]
            }
        },
        "SVR": {
            "model": SVR(),
            "params": {
                "C": [1, 3, 5, 10],
                "epsilon": [0.2, 0.5, 1, 2],
                "kernel": ["rbf"]
            }
        }
    }
    scores = []

    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
        clf.fit(X_train, y_train.values.ravel())
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })

    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print(df)

    ## Возьмем за основу RandomForest
    model = RandomForestRegressor(max_depth=4, n_estimators=200)
    model.fit(X_train, y_train.values.ravel())
    print(model.predict(X_test))

    ## Теперь сравним предсказанные значения с реальными
    print(np.mean(np.abs(y_test.values.ravel()-model.predict(X_test))))


def matrix_ratio_DL_test():
    X = X_bp_no_MinMax.drop("Соотношение матрица-наполнитель", axis="columns")
    y = X_bp_no_MinMax["Соотношение матрица-наполнитель"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    import keras
    from keras import layers
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse'])

    model.fit(X_train, y_train, epochs=100)
    print(model.evaluate(X_test, y_test))
    print(model.predict(X_test))

    first_try = model.fit(X_train, y_train, epochs=100)

    ## Попробуем в модель заложить DropOut
    ## Таким образом попытаемся уменьшить переобучение

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])


    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['mae'])

    model.fit(X_train, y_train, epochs=200)
    print(model.evaluate(X_test, y_test))
    print(model.predict(X_test))

    second_try = model.fit(X_train, y_train, epochs=200)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])


    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse'])

    model.fit(X_train, y_train, epochs=200)
    print(model.evaluate(X_test, y_test))
    print(model.predict(X_test))

    third_try = model.fit(X_train, y_train, epochs=200)

    ## Создадим график

    plt.plot(first_try.history["mse"], label = "first_mse")
    plt.plot(second_try.history["mae"], label="second_mae")
    plt.plot(third_try.history["mse"], label="third_mse")
    plt.ylim([0,0])
    plt.xlabel("Количество epochs")
    plt.ylabel("Соотношение матрица-наполнитель error's")
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Первая попытка создания нейронной сети имеет наименьшее число loss. Используем её как основу


## ????

def score_check():
    X = X_bp_no_MinMax.drop("Соотношение матрица-наполнитель", axis="columns")
    y = X_bp_no_MinMax["Соотношение матрица-наполнитель"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


    import keras
    from keras import layers
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse'])

    model.fit(X_train, y_train, epochs=200)
    print(model.evaluate(X_test, y_test))

    r = LinearRegression()
    r.fit(X_train, y_train)
    res = r.predict(X_test)
    print(y_test)
    print(res)
    print(r2_score(y_test, np.ravel(res)))

##model.save("Папка куда необходимо сохранить модель")
## Сохранение модели через keras

##model = keras.models.load_model('Папка откуда загружается модель')
## Загрузка модели через keras
