#here we will write the code to save our model

import torch
from pathlib import Path
def save_model(model:torch.nn.Module,
target_dir:str,
model_name:str):
  target_dir_path=Path(target_dir)
  target_dir_path.mkdir(parents=True,exist_ok=True)

  #create model save path
  assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name end with '.pt' or '.pth; "
  model_save_path=target_dir_path/model_name
  #save the model state dict
  print(f'[Info] saving model to : {model_save_path}')
  torch.save(obj=model.state_dict(),
  f=model_save_path)