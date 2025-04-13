from typing import NamedTuple, cast
import numpy as np

import os
from PIL import Image


class Box(NamedTuple):
    left: int
    top: int
    width: int
    height: int


def filter_rgb_color_range(img: np.ndarray, p0: tuple[int, int, int], p1: tuple[int, int, int]) -> np.ndarray:
    masks = [
        (img >= p0[i]) & (img <= p1[i])
        for i in range(3)
    ]
    return cast(np.ndarray, np.all(masks, axis=0))


def find_structure_points(arr: np.ndarray, x0: int, y0: int) -> list[tuple[int, int]]:
    # NOTE: does modify arr in place
    structure_points: list[tuple[int, int]] = []
    fringe = [(x0, y0)]
    while len(fringe):
        (x, y) = fringe.pop()
        if arr[x, y]:
            arr[x, y] = False
            structure_points.append((x, y))
            new_x_vals = [x-1, x, x+1]
            new_y_vals = [y-1, y, y+1]
            for new_x in new_x_vals:
                if new_x < 0 or new_x >= arr.shape[0]:
                    continue
                for new_y in new_y_vals:
                    if new_y < 0 or new_y >= arr.shape[1]:
                        continue
                    if arr[new_x, new_y]:
                        fringe.append((new_x, new_y))
    return structure_points


def points_to_binary_image(points: list[tuple[int, int]]) -> tuple[Box, np.ndarray]:
    x0 = min(points, key=lambda p: p[0])[0]
    y0 = min(points, key=lambda p: p[1])[1]
    x1 = max(points, key=lambda p: p[0])[0]
    y1 = max(points, key=lambda p: p[1])[1]
    image_position = Box(x0, y0, x1-x0+1, y1-y0+1)
    image = np.zeros((x1-x0+1, y1-y0+1), dtype=bool)
    for x, y in points:
        image[x-x0, y-y0] = True
    return image_position, image


def find_connected_disjoint_structures(arr: np.ndarray) -> list[tuple[Box, np.ndarray]]:
    structures: list[tuple[Box, np.ndarray]] = []
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x, y]:
                points = find_structure_points(arr, x, y)
                structures.append(points_to_binary_image(points))
    return structures


def pad_combine(images: list[np.ndarray]) -> np.ndarray:
    shape: tuple[int, int] = (
        max([i.shape[0] for i in images]), max([i.shape[1] for i in images]))
    return np.stack([np.pad(img, ((0, shape[0]-img.shape[0]), (0, shape[1]-img.shape[1]))) for img in images])


def cluster_images(images: list[np.ndarray], num_clusters: int):
    # assert len(images) >= num_clusters
    combined = pad_combine(images).reshape(len(images), -1)
    from sklearn.cluster import KMeans
    cluster_algorithm = KMeans(
        n_clusters=num_clusters, max_iter=1000, tol=1e-5)
    return cluster_algorithm.fit_predict(combined)


if __name__ == '__main__':
    from zipfile import ZipFile
    split_images: list[np.ndarray] = []
    if not os.path.exists('edges'):
        os.mkdir('edges')
    with ZipFile('images.zip', 'r') as F:
        # for name in [F.namelist()[-1]]:
        for name in F.namelist():
            if name.endswith('.png'):
                image: np.ndarray = np.array(
                    Image.open(F.open(name)).convert('RGB'))
                image = filter_rgb_color_range(
                    image, (160, 154, 157), (255, 255, 255))
                # image[image > 0] = 255
                # cv2.imwrite(f'edges/{name.split("/")[-1]}', image)
                image = np.all(image != 0, -1)
                Image.fromarray(image).save(f'edges/{name.split("/")[-1]}')
                # image = cv2.Canny(image, 100, 200)
                for symbol in find_connected_disjoint_structures(image):
                    split_images.append(symbol[1])
    labels = cluster_images(split_images, 20)  # 10 numerals, plus commas
    if not os.path.exists('processed_images'):
        os.mkdir('processed_images')
    for i in range(20):
        if not os.path.exists(f'processed_images/{i}'):
            os.mkdir(f'processed_images/{i}')
    for idx, (x, y) in enumerate(zip(labels, split_images)):
        Image.fromarray(y).save(
            f'processed_images/{x}/{idx}.png')
