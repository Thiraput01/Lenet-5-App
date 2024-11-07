import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import LeNet_5

writer = SummaryWriter('runs/mnist')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
batch_size = 128
num_epochs = 50

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the images to 32x32
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.RandomRotation(10),  # Randomly rotate the images by 10 degrees
    transforms.ToTensor(),  # Convert the images to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images
])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the images to 32x32
    transforms.ToTensor(),  # Convert the images to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=train_transform, download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=val_transform)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
sample, labels = next(examples)
sample.shape, labels.shape

image = sample[0]  # Take the first image from the batch
writer.add_image('mnist_images', image, dataformats='CHW')

model = LeNet_5().to(device)
writer.add_graph(model, sample.to(device))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', loss.item(), global_step= i + len(train_loader) * epoch)
            writer.add_scalar('training accuracy', 100 * (outputs.argmax(1) == labels).sum().item() / batch_size, global_step= i + len(train_loader) * epoch)
            writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step= i + len(train_loader) * epoch)
    
    return total_loss / len(train_loader)

def val_one_epoch(model, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for j, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Val accuracy: {accuracy} %')
        writer.add_scalar('val accuracy', accuracy, global_step= len(train_loader) * epoch + j)

    print('--------------------------')
    return accuracy

# Train the model
best_accuracy = 0.0
for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    val_accuracy = val_one_epoch(model, val_loader, epoch)
    
    # Save the model if the validation accuracy is the best we've seen so far.
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_lenet5.pth')
        print(f'Saved Best Model with accuracy: {best_accuracy} %')

writer.flush()
writer.close()

# Test the model
model.load_state_dict(torch.load('best_lenet5.pth'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')
