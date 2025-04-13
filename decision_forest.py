from sklearn.tree import DecisionTreeClassifier
from typing import Literal, cast
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
from image_manipulate_utils import pad_combine
from pickle import dump
data: list[tuple[np.ndarray, int]] = []
y: int | Literal['comma']
for y in cast(list[int | Literal['comma']], [*range(10), 'comma']):
    files: list[str] = os.listdir(f'processed_images/{y}')
    for file in files:
        image: np.ndarray = np.array(
            Image.open(f'processed_images/{y}/{file}'))
        data.append((image, 10 if y == 'comma' else y))
X: np.ndarray = pad_combine([i[0] for i in data]).reshape(len(data), -1)
Y: list[int] = [i[1] for i in data]
X_train: np.ndarray
X_test: np.ndarray
Y_train: list[int]
Y_test: list[int]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, X_test.shape, len(Y_train), len(Y_test))
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, Y_train)
preds = forest.predict_proba(X_test)
prob_error = (1 - preds[Y_test]).mean()
print(prob_error)
print((forest.predict(X_test) == Y_test).mean())
print((preds.argmax(axis=1) == Y_test).mean())
print((1 - forest.predict_proba(X_train)[Y_train]).mean())
print((forest.predict(X_train) == Y_train).mean())
forest.get_params()
with open('decision_forest.pkl', 'wb') as f:
    dump(forest, f)
