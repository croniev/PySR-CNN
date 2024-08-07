import numpy as np
from model import CNN
from torchvision import datasets
from torchvision.transforms import ToTensor
from pysr import PySRRegressor
import torch

# Load model
cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pt'))
print('Model loaded succesfully!')
print(cnn)

# Get Kernels
# print(f"Kernels of first layer: ")
# for name, param in cnn.named_parameters():
#     if name == 'conv1.0.weight':
#         kernels1 = param
#         break
# print(kernels1)

# Load dataset
test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(),)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True, num_workers=1)
samples, labels = next(iter(test_loader))

# Model results
cnn.eval()
with torch.no_grad():
    results = cnn(samples)

# Inputs for PySR (Kernel 1)
# Extract the 5x5 submatrices (incl. padding) from images to use them as input
kernel_size = 5
X = None
for x in samples:
    # x = samples[0][0]
    x = torch.nn.functional.pad(input=x[0], pad=(2, 2, 2, 2), mode="constant", value=0)
    for i, j in np.ndindex((x.size()[0] - kernel_size + 1, x.size()[1] - kernel_size + 1)):
        slice = x[i:i + kernel_size, j:j + kernel_size]
        if X is None:
            X = np.array([slice.numpy().flatten()])
        else:
            X = np.concatenate((X, [slice.numpy().flatten()]))

print(X.shape)

y = results['relu1'][:, 0].numpy().flatten()
print(y.shape)
print(y)

# Get results from PySR
regr = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "square",
        "cube",
        "inv(x) = 1/x",  # Julia syntax
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},  # Sympy syntax
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",  # Julia syntax
)

regr.fit(X, y)
print(regr)
