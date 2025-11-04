import torch
from torchvision import datasets, transforms
from tqdm import tqdm # Thư viện để tạo thanh tiến trình



#  Định nghĩa hàm để tải dữ liệu vào list
def load_data_to_lists(dataset):
    """
    Hàm này duyệt qua toàn bộ dataset và tải ảnh, nhãn vào RAM.
    - Ưu điểm: Dễ dàng truy cập ngẫu nhiên sau khi đã tải xong.
    - Nhược điểm: Tốn rất nhiều RAM và thời gian tải ban đầu.
    """
    images = []
    labels = []
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for img, label in tqdm(dataset, desc=f"Đang tải {len(dataset)} ảnh"):
        # `img` đã là tensor nhờ `transform.ToTensor()`
        # `label` là một số nguyên, ta chuyển nó thành tensor
        images.append(img)
        labels.append(torch.tensor(label, dtype=torch.long))
        
    return images, labels

