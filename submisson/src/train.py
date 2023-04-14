import os
import numpy as np
from tqdm import tqdm

import torch

import src.parameter as PARAM

from src.loss import MSE
from src.model import UNet
from src.dataloader import create_generators

# Fix seed for reproducibility
seed = 123
torch.manual_seed(seed)


def train():
    
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    train_generator, val_generator, _ = create_generators()
    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)

    model.to(device)
    # summary(model, input_size=(3, 256, 256))

    # Loss
    criterion = MSE([8, 9, 11]) #on rajoute la neige aussi
    #criterion = nn.MSELoss()
    # Loss
    #criterion = nn.SmoothL1Loss() #on utilise la SmoothL1Loss qui est comme mean-squarred error mais en mieux

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM.LEARNING_RATE)

    # Metrics
    # metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
    # logging_path = train_logger(config, metrics_name)

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(1, PARAM.NB_EPOCHS + 1):
      print('epoch:' + str(epoch))
      train_loss = []
      model.train()
      train_range = tqdm(train_generator)
      
      for (X, Y) in train_range:
          X = X.to(torch.float).to(device)
          Y,Ybis = Y[:,0,:,:].to(torch.float).to(device),Y[:,1,:,:].to(torch.float).to(device) 
          optimizer.zero_grad()
          S2_pred = model(X)
          loss = criterion(S2_pred, Y,Ybis) 
          loss.backward()
          optimizer.step()
          train_loss.append(loss.item())
          train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
          train_range.refresh()
          
      train_losses.append(np.mean(train_loss))

      ###############################################################
      # Start Evaluation                                            #
      ###############################################################

      model.eval()
      val_loss = []
      with torch.no_grad():
          for (image, target) in tqdm(val_generator, desc='validation'):
              image = image.to(device).to(torch.float)
              y_true = target.to(device).to(torch.float)
              y_true,y_bis = y_true[:,0,:,:].to(torch.float).to(device),y_true[:,1,:,:].to(torch.float).to(device)
              y_pred = model(image)

              loss = criterion(y_pred, y_true,y_bis)
              val_loss.append(loss.item())

      val_losses.append(np.mean(val_loss))
      torch.save(model.state_dict(), os.path.join(PARAM.PATH, 'UNET_' + str(epoch) + '.pth'))

    # Plot losses
    # epochs = range(1, PARAM.NB_EPOCHS + 1)
    # plt.plot(epochs, train_losses, label='train')
    # plt.plot(epochs, val_losses, label='val')