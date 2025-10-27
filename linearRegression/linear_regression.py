import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. data generation
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).unsqueeze(1)
y = 2 * X + 3 + torch.randn(X.size()) * 2

# 2. model define (Linear Regression)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegressionModel()

# 3. loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


def visualization():

    # 5. visualization
    predicted = model(X).detach()

    plt.scatter(X.numpy(), y.numpy(), label="data")
    plt.plot(X.numpy(), predicted.numpy(), color="red", label="Fitted Line")
    plt.legend()
    plt.show()

# 4. loop
epochs = 200
for epoch in range(epochs):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        visualization()
