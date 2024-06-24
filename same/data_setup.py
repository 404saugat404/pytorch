#here we will create datasets and data loaders for our task
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_of_workers=os.cpu_count()

def create_data_loaders(
  train_dir:str,
  test_dir:str,
  transform:transforms.Compose,
  batch_size:int,
  num_workers:int=num_of_workers
  ):
  #using ImageFolder to create datasets
  train_data=datasets.ImageFolder(train_dir,transform=transform)
  
  test_data=datasets.ImageFolder(test_dir,transform=transform)

  #getting class names:
  class_names=train_data.classes

  #turning images to data loaders
  train_dataloaders=DataLoader(
    train_data,
    batch_size,
    shuffle=True,
    num_workers=num_of_workers,
    pin_memory=True
  )
  test_dataloaders=DataLoader(
    test_data,
    batch_size,
    shuffle=False,
    num_workers=num_of_workers,
    pin_memory=True
  )

  return train_dataloaders, test_dataloaders, class_names