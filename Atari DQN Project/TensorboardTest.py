
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


# Example on how to use .add_scalar to measure loss in simple linear regression training
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("Starting training")
train_model(10)
writer.flush()
print("Done")
# writer.close() # do this if we do not need the summary writer anymore


