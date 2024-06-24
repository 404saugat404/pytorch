import os
import requests
import zipfile
from pathlib import Path

#setting up path to data folder
data_path=Path('data/')
image_path=data_path/"pizza_steak_sushi"
if image_path.is_dir():
  print(f'{image_path} directry exists')

else:
  print(f'creating {image_path} directory')
  image_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request=requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print('downloading ...')
  f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
