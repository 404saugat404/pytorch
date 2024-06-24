#here we will create training and testing steps
from IPython.core.displayhook import Float
import torch
from torchvision.datasets.vision import data
from tqdm.auto import tqdm
from typing import Dict,List,Tuple
def train_step(
  model:torch.nn.Module,
  dataloader:torch.utils.data.dataloader,
  loss_fn:torch.nn.Module,
  optimizer:torch.optim.Optimizer,
  device:torch.device)-> Tuple[Float,Float]:

  for batch,(X,y) in enumerate(dataloader):
    X,y=X.to(device),y.to(device)

    #forward pass
    y_pred=model(X)

    #calculate and accumulate loss
    loss=loss_fn(y_pred, y)
    train_loss=train_loss+loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
    train_acc+=(y_pred_class==y).sum().item()/len(y_pred)

  #adjusting metrics to get avg loss and accuracy per batch
  train_loss=train_loss/len(dataloader)
  train_acc=train_acc/len(dataloader)

  return train_loss, train_acc

  def test_step(
  model:torch.nn.Module,
  dataloader:torch.utils.data.dataloader,
  loss_fn:torch.nn.Module,
  device:torch.device)-> Tuple[Float,Float]:
  
    model.eval()

    test_loss,test_acc=0,0
    with torch.inference_mode():
      for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        #forward pass
        test_pred_logits=model(X)

        #calculate and accumulate loss
        loss=loss_fn(test_pred, y)
        test_loss=train_loss+loss.item()

        test_pred_labels=test_pred_logits.argmax(dim=1)
        test_acc+=((test_pred_labels==y).sum().item()/len(test_pred_labels))

      #adjusting metrics to get avg loss and accuracy per batch
      test_loss=test_loss/len(dataloader)
      test_acc=test_acc/len(dataloader)

    return test_loss, test_acc


  def train(
    model:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    
    test_dataloader:torch.utils.data.DataLoader,
    optimizer:torch.optim.Optimizer,
    loss_fn:torch.nn.Module,
    epochs:int,
    device:torch.device
    )->Dict[str,list]:
    
    results={"train_loss":[],
    'train_acc':[],
    'test_loss':[],
    'test_acc':[]
    }

    for epoch in tqdm(range(epochs)):
      train_loss,train_acc=train_step(model=model,
      dataloader=train_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      device=device)

      
      test_loss,test_acc=test_step(model=model,
      dataloader=test_dataloader,
      loss_fn=loss_fn,
      device=device)

      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results

    
