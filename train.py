from data_loader import load_data_to_lists
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset # <-- Thêm TensorDataset vào
import torch
from modal import ModalEncoder
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sử dụng thiết bị: {device}')

    # Chỉ cần định nghĩa transform MỘT LẦN cho ảnh 48x48
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh màu -> ảnh xám
        transforms.Resize((48, 48)),                  # Đảm bảo đúng kích thước
        transforms.ToTensor(),                        # Chuyển sang tensor [0, 1]
        transforms.Normalize((0.5,), (0.5,))          # Chuẩn hóa về khoảng [-1, 1] cho ảnh xám
    ])

    # 1. Tải dataset từ thư mục
    train_dataset_folder = datasets.ImageFolder(root='data/train', transform=transform)
    
    # 2. Sử dụng hàm của bạn để tải dữ liệu vào RAM
    #    Hàm này trả về 2 list: một list chứa các tensor ảnh và một list chứa các tensor nhãn
    train_images_list, train_labels_list = load_data_to_lists(train_dataset_folder)

    # 3. Chuyển 2 list thành 2 tensor lớn
    #    torch.stack sẽ gộp các tensor trong list thành một tensor duy nhất
    train_images_tensor = torch.stack(train_images_list)
    train_labels_tensor = torch.stack(train_labels_list)

    # 4. Tạo một Dataset từ các tensor này
    final_train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    
    # 5. Tạo DataLoader từ Dataset đã tạo
    #    Lưu ý: Bây giờ chỉ cần truyền Dataset và các tham số cấu hình
    train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True)

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