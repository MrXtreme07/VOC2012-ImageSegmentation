import torch
from models.unet import UNet
from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset
import torch.nn as nn
import torch.optim as optim
import os

os.makedirs('checkpoints', exist_ok=True)

train_dataset = VOCDataset('dataset/VOC2012', 'train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = torch.device('cpu')
model = UNet(num_classes=21).to(device)

start_epoch = 150
end_epoch = 200

checkpoint_path = f'checkpoints/model_epoch_{start_epoch}.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
# start_epoch = checkpoint['epoch'] + 1

model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Loaded checkpoint: ', checkpoint_path)

for epoch in range(start_epoch, end_epoch):
    model.train()
    total_loss = 0
    print(f'Starting epoch {epoch+1}/{end_epoch}...')

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} Loss: {avg_loss:.4f}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoints/model_epoch_{epoch+1}.pth')