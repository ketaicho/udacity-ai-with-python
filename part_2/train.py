# PROGRAMMER: Tshepo Molebiemang
# DATE: 10 November 2021
import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import argparse
from pathlib import Path
from workspace_utils import active_session

def get_transforms():

    trans_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     trans_norm]),
        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    trans_norm]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     trans_norm])
    }

    return data_transforms


def get_datasets(data_transforms, data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    return image_datasets


def get_data_loaders(image_datasets):

    data_loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False)
    }

    return data_loaders


def classifier(arch, out_features, hidden_units):

    model = getattr(models, arch)(pretrained=True)

    if arch == 'densenet121':
        in_features = model.classifier.in_features
    else:
        in_features = model.classifier[0].in_features

    half_hidden = int(hidden_units * 0.5)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(in_features, hidden_units)),
                                ('relu1', nn.ReLU()),
                                ('drop1', nn.Dropout(p = 0.2)),
                                ('fc2', nn.Linear(hidden_units, half_hidden)),
                                ('relu2', nn.ReLU()),
                                ('drop2', nn.Dropout(p = 0.2)),
                                ('fc3', nn.Linear(half_hidden, out_features)),
                                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    return model


def train(model, criterion, optimizer, data_loaders, device):
    
    running_loss = 0
    correct = 0
    total = 0
    
    model.train()

    for images, labels in data_loaders['train']:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(data_loaders['train'])
    accuracy = (100 * correct) / total
    
    print("Training Loss: {:.3f}.. ".format(train_loss),
          "Training Accuracy: {:.3f}.. ".format(accuracy))
    
    return train_loss

def test(model, criterion, data_loaders, device):
    
    running_loss = 0
    correct = 0
    total = 0
    
    model.eval()

    with torch.no_grad():
        for images, labels in data_loaders['test']:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            
            loss = criterion(output, labels)
            running_loss += loss.item()
            
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_loss = running_loss / len(data_loaders['test'])
        accuracy = (100 * correct) / total

    print("Testing Loss: {:.3f}.. ".format(test_loss),
          "Testing Accuracy: {:.3f}.. ".format(accuracy))


def get_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers/', help='path to the flower images folder')
    parser.add_argument('--save_dir', type=str, default=None, help='path to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help="'vgg' or 'densenet'")
    parser.add_argument('--device', type=str, default='gpu', help="'cpu' or 'gpu'")
    parser.add_argument('--epochs', type=int, default=10, help="no. times to run training/testing loops")
    parser.add_argument('--hidden_units', type=int, default=2048, help="no. of hidden units")
    parser.add_argument('--learning_rate', type=float, default=0.01)

    return parser.parse_args()


def create_checkpoint(path, model, optimizer, arch, num_epochs, hidden_units, image_datasets):

    model.class_to_idx = image_datasets['train'].class_to_idx

    torch.save({'arch': arch,
                'epochs': num_epochs,
                "hidden_units": hidden_units,
                'class_to_idx': model.class_to_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)

    print("Checkpoint created!")


def main():
    args = get_input_args()

    arch = args.arch
    if arch == 'vgg':
        arch = 'vgg16'
    elif arch == 'densenet':
        arch = 'densenet121'

    if args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # *****************************************
    print(" *** The training parameters are as follows:")

    data_dir = args.data_dir
    print(f"data_dir: {data_dir}")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_path = args.save_dir + '/checkpoint.pth'
    else:
        save_path = 'checkpoint.pth'

    hidden_units = args.hidden_units

    print(f"save_dir: {args.save_dir}")
    print(f"save_path: {save_path}")
    print(f"arch: {arch}")
    print(f"device: {device}")
    print(f"hidden_units: {hidden_units}")

    learning_rate = args.learning_rate
    print(f"learning_rate: {learning_rate}")

    num_epochs = args.epochs
    print(f"num_epochs: {num_epochs} \n")
    # *****************************************

    data_transforms = get_transforms()

    image_datasets = get_datasets(data_transforms, data_dir)

    data_loaders = get_data_loaders(image_datasets)

    out_features = len(image_datasets['train'].class_to_idx)

    model = classifier(arch, out_features, hidden_units)
    if torch.cuda.is_available():
        model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

    with active_session():
        for epoch in range(num_epochs):
            print("Epoch #: {}/{}".format(epoch + 1, num_epochs))

            train(model, criterion, optimizer, data_loaders, device)

            test(model, criterion, data_loaders, device)

            print("\n")

    print("** Training has completed! **")

    create_checkpoint(save_path, model, optimizer, arch, num_epochs, hidden_units, image_datasets)


if __name__ == "__main__":
    main()