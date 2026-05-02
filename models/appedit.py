import tkinter as tk
from tkinter import filedialog
import requests
import os
from PIL import Image, ImageTk

#URL of the api server
API_URL = "http://127.0.0.1:8000/predict"

root = tk.Tk()
root.title("Breast Cancer Classifier")
root.geometry("420x550")
root.configure(bg="white")
#referencing the image
photo_ref = None

def select_image():
    """ oopen file for user to select image then display preview then be able to use predict"""
    global photo_ref
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return
#resize image
    img = Image.open(path)
    img.thumbnail((300, 300))
    photo_ref = ImageTk.PhotoImage(img)
    image_label.config(image=photo_ref)
    result_label.config(text="Click Predict to analyse", fg="gray")
    predict_btn.config(state="normal")
    predict_btn.path = path

def predict():
    """sends the image to the api server and returns back the predicted class and confidence to display result"""
    path = predict_btn.path
    result_label.config(text="Analysing...", fg="gray")
    root.update()

    try:
#send image to api as multipart form data
        ext = os.path.splitext(path)[-1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            response = requests.post(API_URL, files={"file": (os.path.basename(path), f, mime)}, timeout=60)
        data = response.json()

        if response.status_code == 200:
            cls = data["predicted_class"]
            conf = data["confidence"]
            color = "red" if "malign" in cls.lower() else "green"
            result_label.config(text=f"{cls.capitalize()}  —  {conf:.1%} confidence", fg=color)
        else:
            result_label.config(text=f"Error: {data.get('detail', 'unknown')}", fg="red")

    except requests.exceptions.ConnectionError:
        result_label.config(text="Cannot connect — is the server running?", fg="red")
    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="red")
#UI layout
tk.Label(root, text="Ultrasound Classifier", font=("Helvetica", 18, "bold"), bg="white").pack(pady=20)

image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

tk.Button(root, text="Select Image", font=("Helvetica", 12), command=select_image, padx=10, pady=6).pack(pady=8)

predict_btn = tk.Button(root, text="Predict", font=("Helvetica", 12, "bold"), fg="white", bg="#c0392b",
                        command=predict, padx=10, pady=6, state="disabled",
                        activebackground="#a93226", activeforeground="white", bd=0)
predict_btn.pack(pady=4)

result_label = tk.Label(root, text="Select an image to get started", font=("Helvetica", 13),
                        bg="white", fg="gray", wraplength=380)
result_label.pack(pady=20)

root.mainloop()
