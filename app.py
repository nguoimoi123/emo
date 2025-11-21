import customtkinter as ctk
import cv2
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
import torchvision.transforms as transforms
from modal import ModalEncoder
import threading
import os

# --- Cấu hình giao diện ---
ctk.set_appearance_mode("dark")  # Chế độ "dark", "light" hoặc "system"
ctk.set_default_color_theme("blue")  # Chủ đề "blue", "green", "dark-blue"

class EmotionRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Nhận dạng Cảm xúc")
        self.geometry("1000x600")

        # --- 1. Cấu hình và tải mô hình (tương tự cam.py) ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {self.device}")

        # Tải mô hình YOLOv8
        try:
            self.yolo_model = YOLO('modal/yolov8n-face-lindevs.pt')
        except Exception as e:
            print(f"Lỗi khi tải mô hình YOLO: {e}")
            self.show_error("Không thể tải mô hình YOLO. Vui lòng kiểm tra file 'modal/yolov8n-face-lindevs.pt'")
            return

        # Tải mô hình nhận dạng cảm xúc
        self.emotion_model_path = 'modal/modal_encoder.pth'
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        try:
            self.emotion_model = ModalEncoder(num_class=len(self.class_names)).to(self.device)
            self.emotion_model.load_state_dict(torch.load(self.emotion_model_path, map_location=self.device))
            self.emotion_model.eval()
            print("Đã tải mô hình cảm xúc thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình cảm xúc: {e}")
            self.show_error(f"Không thể tải mô hình cảm xúc. Vui lòng kiểm tra file '{self.emotion_model_path}'")
            return

        # Định nghĩa transform cho ảnh khuôn mặt
        self.emotion_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # --- 2. Khởi tạo các biến ---
        self.cap = None
        self.camera_thread = None
        self.is_camera_running = False

        # --- 3. Tạo giao diện ---
        self.setup_ui()

    def setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=3)  # Cột trái (video) chiếm 3/4 không gian
        self.grid_columnconfigure(1, weight=1)  # Cột phải (nút bấm) chiếm 1/4
        self.grid_rowconfigure(0, weight=1)

        # Frame bên trái để hiển thị video/ảnh
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Nhấn 'Mở Camera' hoặc 'Chọn Ảnh' để bắt đầu")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Frame bên trái để hiển thị video/ảnh
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.control_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)

        # Tiêu đề
        self.title_label = ctk.CTkLabel(self.control_frame, text="Điều khiển", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=10, pady=10)

        # Nút "Mở Camera"
        self.open_camera_btn = ctk.CTkButton(self.control_frame, text="Mở Camera", command=self.open_camera)
        self.open_camera_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Nút "Chọn Ảnh"
        self.select_image_btn = ctk.CTkButton(self.control_frame, text="Chọn Ảnh", command=self.select_image)
        self.select_image_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Nút "Dừng"
        self.stop_btn = ctk.CTkButton(self.control_frame, text="Dừng", command=self.stop_camera, fg_color="red", hover_color="darkred")
        self.stop_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Nhãn trạng thái
        self.status_label = ctk.CTkLabel(self.control_frame, text="Sẵn sàng", wraplength=150)
        self.status_label.grid(row=4, column=0, padx=20, pady=10)

    def process_frame(self, frame):
        """Xử lý một khung hình: phát hiện khuôn mặt và nhận dạng cảm xúc."""
        results = self.yolo_model(frame, verbose=False)
        result = results[0]
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])

            # YOLOv8 face-lindevs chỉ có một lớp 'face'
            if confidence > 0.5:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_tensor = self.emotion_transform(face_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.emotion_model(face_tensor)
                    _, predicted_idx = torch.max(output, 1)
                    predicted_class_name = self.class_names[predicted_idx.item()]

                label = f'{predicted_class_name}: {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

    def update_camera(self):
        """Cập nhật khung hình từ camera và hiển thị lên giao diện."""
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                
                # Chuyển đổi khung hình OpenCV (BGR) sang PIL Image (RGB)
                image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
                # Chuyển đổi PIL Image sang ImageTk để hiển thị trong CTkLabel
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(800, 600))
                self.video_label.configure(image=ctk_image, text="")
                
                # Lặp lại việc cập nhật sau 10ms
                self.after(10, self.update_camera)
            else:
                self.stop_camera()
                self.status_label.configure(text="Lỗi: Không thể đọc khung hình từ camera.")
        else:
            # Nếu camera không chạy, không làm gì cả
            pass

    def open_camera(self):
        """Mở camera và bắt đầu luồng video."""
        if self.is_camera_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Lỗi: Không thể mở webcam.")
            return
        
        self.is_camera_running = True
        self.status_label.configure(text="Đang chạy camera...")
        self.update_camera()

    def stop_camera(self):
        """Dừng camera và giải phóng tài nguyên."""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_label.configure(text="Đã dừng.")
        # Xóa ảnh hiện tại
        self.video_label.configure(image="", text="Nhấn 'Mở Camera' hoặc 'Chọn Ảnh' để bắt đầu")

    def select_image(self):
        """Mở hộp thoại để chọn ảnh và xử lý nó."""
        self.stop_camera() # Dừng camera trước khi xử lý ảnh
        
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Chọn một ảnh",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            self.status_label.configure(text="Đang xử lý ảnh...")
            frame = cv2.imread(file_path)
            if frame is None:
                self.show_error("Lỗi: Không thể đọc file ảnh.")
                self.status_label.configure(text="Sẵn sàng")
                return

            processed_frame = self.process_frame(frame)
            
            # Chuyển đổi và hiển thị ảnh
            image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(800, 600))
            self.video_label.configure(image=ctk_image, text="")
            self.status_label.configure(text="Đã xử lý xong.")

    def show_error(self, message):
        """Hiển thị hộp thoại lỗi."""
        ctk.CTkMessageBox(master=self, title="Lỗi", message=message)

    def on_closing(self):
        """Hàm được gọi khi đóng cửa sổ."""
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = EmotionRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Đảm bảo dừng camera khi đóng cửa sổ
    app.mainloop()