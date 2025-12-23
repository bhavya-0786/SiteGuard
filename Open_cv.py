import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import threading
import time

class ConstructionApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("SiteGuard")
        self.root.geometry("1000x700")

        # Background
        bg_path = "/Users/bhavya/Downloads/pexels-umaraffan499-190417.jpg"
        bg_image = Image.open(bg_path).resize((1000, 700))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load YOLO model
        self.model = YOLO(model_path)
        
        title = tk.Label(root, text="Construction Site Object Detection", 
                         font=("Times New Roman", 40),  fg="white")
        title.pack(pady=20)        

        # GUI panel
        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(side="bottom", pady=15)

        self.start_btn = tk.Button(btn_frame, text="Start detection", command=self.start_camera,
                                   bg="green", fg="black", width=15)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="Stop detection", command=self.stop_camera,
                                  bg="red", fg="black", width=15)
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.quit_btn = tk.Button(btn_frame, text="Exit", command=self.on_exit,
                                  bg="black", fg="black", width=15)
        self.quit_btn.grid(row=0, column=2, padx=10)

        self.cap = None
        self.running = False
        self.last_beep_time = 0
        self.frame = None
        self.annotated = None

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam")
                return
            self.running = True
            threading.Thread(target=self.detect_loop, daemon=True).start()
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.panel.config(image="")

    def on_exit(self):
        self.stop_camera()
        self.root.destroy()

    def play_beep(self):
        os.system('afplay /System/Library/Sounds/Submarine.aiff')

    def play_beep_async(self):
        threading.Thread(target=self.play_beep, daemon=True).start()

    def detect_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame = frame.copy()
            # YOLO detection
            results = self.model(frame, imgsz=416, conf=0.5, verbose=False)
            self.annotated = results[0].plot()

            # Extract class names
            class_names = [self.model.names[int(cls)] for cls in results[0].boxes.cls]

            # Danger detection
            if "head" in [c.lower() for c in class_names]:
                # Draw visual warning on the annotated frame
                cv2.putText(self.annotated, "DANGER!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                # Optional beep (non-blocking)
                current_time = time.time()
                if current_time - self.last_beep_time > 2:  # 2-second cooldown
                    self.play_beep_async()
                    self.last_beep_time = current_time

    def update_frame(self):
        if self.running:
            if self.annotated is not None:
                # Convert annotated frame for Tkinter
                annotated_rgb = cv2.cvtColor(self.annotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
            self.root.after(15, self.update_frame)  # smooth ~60 FPS

if __name__ == "__main__":
    MODEL_PATH = "/Users/bhavya/Downloads/best_model.pt"
    root = tk.Tk()
    app = ConstructionApp(root, MODEL_PATH)
    root.mainloop()
