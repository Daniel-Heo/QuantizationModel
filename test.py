# 모델 정의 및 훈련
# 간단한 모델을 정의하고 훈련합니다. 여기서는 예시로 분류 문제를 위한 간단한 신경망 모델을 사용합니다.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# 모델 인스턴스화 및 훈련 설정
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# -------------------------------------
# 모델 동적 양자화
# 훈련된 모델을 동적 양자화합니다. PyTorch에서는 torch.quantization.quantize_dynamic 함수를 사용하여 이 과정을 간단히 수행할 수 있습니다.

model.eval()  # 모델을 평가 모드로 설정

# 모델 동적 양자화
quantized_model = torch.quantization.quantize_dynamic(
    model,  # 양자화할 모델
    {nn.Linear},  # 양자화할 레이어 유형
    dtype=torch.qint8  # 가중치의 데이터 타입
)

# 양자화된 모델의 크기 확인 (예시)
print(quantized_model)

# -------------------------------------
# 양자화된 모델 사용
# 양자화된 모델을 사용하여 추론을 수행합니다. 양자화된 모델의 사용법은 원본 모델과 동일합니다.

# 추론을 위한 데이터 준비
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 양자화된 모델로 추론
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(f'Predicted: {predicted.item()}, Actual: {labels.item()}')
        break  # 예시이므로 하나의 샘플에 대해서만 추론을 수행