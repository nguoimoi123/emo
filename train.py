from data_loader import load_data_to_lists
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from modal import ModalEncoder
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sử dụng thiết bị: {device}')

    transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh màu -> ảnh xám
    transforms.Resize((48, 48)),                  # Đảm bảo đúng kích thước
    transforms.ToTensor(),                        # Chuyển sang tensor [0, 1]
    transforms.Normalize((0.5,), (0.5,))          # Chuẩn hóa về khoảng [-1, 1]
])

    train_data = datasets.ImageFolder(root='data/train', transform=transform_train)
    train_data, train_labels = load_data_to_lists(train_data)

    # Tiền xử lý dữ liệu thành DataLoader
    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Tạo DataLoader từ danh sách ảnh và nhãn
    train_loader = DataLoader(train_data, transforms, batch_size=32, shuffle=True)

    # Khởi tạo mô hình, hàm mất mát và bộ tối ưu hóa
    model = ModalEncoder(num_class=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


    # Vòng lặp huấn luyện
    num_epochs = 100
    num_iterations = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            #Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward và tối ưu hóa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_iterations}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {running_loss/num_iterations:.4f}')

# Lưu mô hình sau khi huấn luyện xong
    torch.save(model.state_dict(), 'modal/modal_encoder.pth')

    print('Đã lưu mô hình vào modal/modal_encoder.pth')