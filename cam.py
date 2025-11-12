import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
from modal import ModalEncoder
import argparse

def main():
    # --- 1. Cấu hình và tải mô hình ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Tải mô hình YOLOv8 để phát hiện khuôn mặt
    # Bạn có thể thay 'yolov8n.pt' bằng đường dẫn đến mô hình YOLO của riêng bạn
    # 'yolov8n.pt' là mô hình nhỏ nhất, nhanh nhất
    try:
        yolo_model = YOLO('modal/yolov8n-face-lindevs.pt')
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLO: {e}")
        print("Vui lòng đảm bảo bạn đã cài đặt ultralytics: pip install ultralytics")
        return

    # Tải mô hình nhận dạng cảm xúc của bạn
    emotion_model_path = 'modal/modal_encoder.pth'
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] # QUAN TRỌNG: Đúng thứ tự
    
    try:
        emotion_model = ModalEncoder(num_class=len(class_names)).to(device)
        emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=device))
        emotion_model.eval()
        print("Đã tải mô hình cảm xúc thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình cảm xúc: {e}")
        return

    # Định nghĩa transform cho ảnh khuôn mặt (phải giống lúc huấn luyện)
    emotion_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- 2. Mở webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return

    print("Đã mở webcam. Nhấn 'q' để thoát.")

    # --- 3. Vòng lặp chính ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ webcam.")
            break

        # Sử dụng YOLO để phát hiện đối tượng
        # verbose=False để không in kết quả ra màn hình console
        results = yolo_model(frame, verbose=False)

        # Lấy kết quả từ khung hình đầu tiên
        result = results[0]
        
        # Duyệt qua các đối tượng được phát hiện
        for box in result.boxes:
            # Lấy tọa độ, độ tin cậy và lớp của đối tượng
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])

            # YOLOv8 được huấn luyện trên dataset COCO, lớp 'person' có id là 0
            # Chúng ta chỉ xử lý nếu phát hiện là người và độ tin cậy cao
            if cls == 0 and confidence > 0.5:
                # Cắt vùng ảnh chứa khuôn mặt
                face_crop = frame[y1:y2, x1:x2]

                # Kiểm tra xem vùng cắt có hợp lệ không
                if face_crop.size == 0:
                    continue

                # Chuyển ảnh từ BGR (OpenCV) sang RGB (PIL)
                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                
                # Tiền xử lý và dự đoán cảm xúc
                face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = emotion_model(face_tensor)
                    _, predicted_idx = torch.max(output, 1)
                    predicted_class_name = class_names[predicted_idx.item()]

                # Vẽ hộp và nhãn lên khung hình
                label = f'{predicted_class_name}: {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị khung hình kết quả
        cv2.imshow('Emotion Recognition with YOLO', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 4. Dọn dẹp ---
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng chương trình.")

if __name__ == '__main__':
    main()