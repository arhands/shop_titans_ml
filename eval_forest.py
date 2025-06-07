from argparse import ArgumentParser
import numpy as np
from PIL import Image
from pickle import load
import pyscreeze
from sklearn.ensemble import RandomForestClassifier

from model import Forest
# process arguments
parser = ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-p', '--path', type=str, default=None)
group.add_argument('-i', '--input_coords_file_path', type=str, default=None,
                   help='path to input coords file. File should be comma delimited as left, top, width, height')
parser.add_argument('-o', '--output_value_path', default=None)
parser.add_argument('-v', '--verbose', default=False)
args = parser.parse_args()
output_value_path: str | None = args.output_value_path
verbose: bool = args.verbose

# loading image
img: Image.Image
if args.path is not None:
  img = Image.open(args.path)
else:
  args_path: str = args.input_coords_file_path
  from pyscreeze import Box, screenshot
  with open(args_path, 'r') as f:
    x, y, w, h = map(int, f.read().split(','))
  img = screenshot(
      region=Box(x, y, w, h))

if verbose:
  img.save('output.png')

# loading decision forest model
model: Forest = Forest.load('decision_forest.pkl')
prediction = model.predict(img)

# removing unneeded characters
prediction = prediction.replace(',', '')
if output_value_path is not None:
  with open(output_value_path, 'w') as f:
    f.write(prediction)
else:
  print(prediction)
