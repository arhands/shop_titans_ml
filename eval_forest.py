from argparse import ArgumentParser
import numpy as np
from PIL import Image
from pickle import load
from sklearn.ensemble import RandomForestClassifier

from image_manipulate_utils import filter_rgb_color_range, find_connected_disjoint_structures
# process arguments
parser = ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-d', '--directory', type=str, default=None)
group.add_argument('-i', '--input_coords_file_path', type=str, default=None,
                   help='path to input coords file. File should be comma delimited as left, top, width, height')
parser.add_argument('-o', '--output_value_path', default=None)
args = parser.parse_args()
output_value_path: str | None = args.output_value_path
img: Image.Image
if args.directory is not None:
    directory: str = args.directory
    img = Image.open(directory)
else:
    args_path: str = args.input_coords_file_path
    from pyscreeze import Box, screenshot
    with open(args_path, 'r') as f:
        x, y, w, h = map(int, f.read().split(','))
    img = screenshot(
        region=Box(x, y, w, h))
img.save('output.png')
image = np.array(img.convert('RGB'))
# loading image
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
res = ''.join([str(p[0]) for p in number_xpos_pairs])
if output_value_path is not None:
    with open(output_value_path, 'w') as f:
        f.write(res)
else:
    print(res)
