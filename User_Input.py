import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

class ImageDetectionApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("SiteGuard")
        self.root.geometry("1000x700")
        self.root.configure(bg="#020101")

        bg_path = "/Users/bhavya/Downloads/pexels-umaraffan499-190417.jpg"  
        bg_image = Image.open(bg_path)
        bg_image = bg_image.resize((1000, 700))
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load YOLO model
        self.model = YOLO(model_path)

        # Title
        title = tk.Label(root, text="Construction Site Object Detection", 
                         font=("Times New Roman", 40),  fg="white")
        title.pack(pady=20)

        # Image display area
        self.image_label = tk.Label(root, bg="#0F0E0E")
        self.image_label.pack(expand=True, pady=10)

        # Bottom frame for buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(side="bottom", pady=25)

        # Buttons
        upload_btn = tk.Button(btn_frame, text="Upload Image", command=self.upload_image, 
                                fg="black", width=15)
        upload_btn.grid(row=0, column=0, padx=10)

        detect_btn = tk.Button(btn_frame, text="Run Detection", command=self.detect_objects, 
                                fg="black", width=15)
        detect_btn.grid(row=0, column=1, padx=10)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_image, 
                              fg="black", width=15)
        clear_btn.grid(row=0, column=2, padx=10)

        exit_btn = tk.Button(btn_frame, text="Exit", command=root.destroy, 
                             fg="black", width=15)
        exit_btn.grid(row=0, column=3, padx=10)

        self.uploaded_image = None
        self.display_image = None

    def upload_image(self):
        """Select image from file system"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        self.uploaded_image = file_path
        img = Image.open(file_path)
        img = img.resize((800, 500))
        imgtk = ImageTk.PhotoImage(img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

    def detect_objects(self):
        """Run YOLO detection on uploaded image"""
        if not self.uploaded_image:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        # Run inference
        results = self.model(self.uploaded_image, imgsz=640, conf=0.5)

        # Annotate and display
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_rgb)
        img = img.resize((800, 500))
        imgtk = ImageTk.PhotoImage(img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

    def clear_image(self):
        """Reset display"""
        self.image_label.config(image="")
        self.image_label.image = None
        self.uploaded_image = None


if __name__ == "__main__":
    MODEL_PATH = "/Users/bhavya/Downloads/best_model.pt"  # same model path as your camera app
    root = tk.Tk()
    app = ImageDetectionApp(root, MODEL_PATH)
    root.mainloop()
