import torch
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from opacus import PrivacyEngine
from model import Net
from torch import nn
from torch.nn import functional as F
# Define the transformation for the dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(".", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
privacy_engine = PrivacyEngine()
model, optimizer, trainloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=trainloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

def train():
    model.train()
    for data, target in trainloader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self): return [val.cpu().numpy() for val in model.parameters()]
    def set_parameters(self, parameters):
        for param, new in zip(model.parameters(), parameters):
            param.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train()
        return self.get_parameters(), len(trainloader.dataset), {}

fl.client.start_numpy_client("localhost:8080", client=FlowerClient())




