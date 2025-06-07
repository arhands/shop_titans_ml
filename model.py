from __future__ import annotations
from typing import Any, Literal
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pickle import dump, load
import numpy as np
from PIL import Image

from image_manipulate_utils import image_to_characters


def _tok_to_index(v: str) -> int:
  array2index: dict[str, int] = {
    ',': 10,
    '/': 11,
  }
  res = array2index.get(v)
  if res is not None:
    return res
  return int(v)


def _index_to_tok(v: int) -> str:
  index2array: dict[int, str] = {
    10: ',',
    11: '/',
  }
  res = index2array.get(v)
  if res is not None:
    return res
  return str(v)


class Forest:
  def __init__(self, n_estimators: int = 100, target_width: int = 48, target_height: int = 48, threshold: int = 200):
    self.forest = RandomForestClassifier(n_estimators=n_estimators)
    self.target_width = target_width
    self.target_height = target_height
    self.threshold = threshold

  def process_image(self, X: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(X, Image.Image):
      X = X.convert('RGB')
    img = np.array(X)
    a = image_to_characters(img, self.target_width,
                            self.target_height, self.threshold)
    return a.reshape(a.shape[0], -1)

  def fit(self, X: np.ndarray, Y: str):
    self.forest.fit(X, [_tok_to_index(y) for y in Y])

  def eval(self, X: np.ndarray, Y: str) -> float:
    return float(self.forest.score(X, [_tok_to_index(y) for y in Y]))

  def predict(self, X: Image.Image) -> str:
    characters = self.process_image(X)
    indices = self.forest.predict(characters)
    return ''.join([_index_to_tok(i) for i in indices])

  @classmethod
  def load(cls, path: str) -> Forest:
    with open(path, 'rb') as f:
      return load(f)

  def save(self, path: str):
    with open(path, 'wb') as f:
      dump(self, f)


if __name__ == '__main__':
  forest = Forest()
  data: dict[Literal['labels', 'images'],
             list[Any]] = load(open('data2.pkl', 'rb'))
  data1 = load(open('data1.pkl', 'rb'))
  images: list[np.ndarray] = list(data['images'])
  images.extend(data1['images'])
  labels: list[str] = data['labels']
  labels.extend(data1['labels'])
  characters = np.concatenate([
    forest.process_image(img)
    for img in images
  ])
  character_labels = ''.join([c for y in labels for c in y])
  X_train, X_test, y_train, y_test = train_test_split(
      characters, character_labels, test_size=0.2, random_state=42)
  forest.fit(X_train, y_train)
  print(forest.eval(X_test, y_test))
  forest.save('decision_forest.pkl')
