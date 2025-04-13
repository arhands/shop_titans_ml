from argparse import ArgumentParser
import numpy as np
from PIL import Image
from pickle import load
from sklearn.ensemble import RandomForestClassifier

from image_manipulate_utils import filter_rgb_color_range, find_connected_disjoint_structures
# process arguments
parser = ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True)
args = parser.parse_args()
path: str = args.directory

# loading image
image: np.ndarray = np.array(Image.open(path).convert('RGB'))
image = filter_rgb_color_range(
    image, (160, 154, 157), (255, 255, 255))
image = np.all(image != 0, -1)

number_xpos_pairs: list[tuple[str, int]] = []

# loading decision forest model
with open('decision_forest.pkl', 'rb') as f:
    forest: RandomForestClassifier = load(f)

for symbol in find_connected_disjoint_structures(image):
    # print(symbol[1].shape)
    X = np.pad(symbol[1], ((0, 17 - symbol[1].shape[0]),
               (0, 10 - symbol[1].shape[1])))
    Y_pred = forest.predict(X.reshape(1, -1))
    if Y_pred[0] == 10:
        # y = ','
        continue  # skipping commas
    else:
        y = str(Y_pred[0])
    number_xpos_pairs.append((y, symbol[0][1]))
    # number_xpos_pairs.append((Y_pred[0], symbol[0][0]))
number_xpos_pairs.sort(key=lambda p: p[1])
print(''.join([str(p[0]) for p in number_xpos_pairs]))
