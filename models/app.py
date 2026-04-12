"""
app.py

Basic Tkinter GUI for submitting an image to the prediction API.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image, ImageTk


API_URL = "http://127.0.0.1:8000/predict"


def select_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")],
    )
    if not file_path:
        return

    # Show the selected image
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

    result_label.config(text="Predicting...")
    root.update()

    # Send to API
    try:
        with open(file_path, "rb") as f:
            response = requests.post(API_URL, files={"image": f}, timeout=60)

        data = response.json()

        if response.status_code == 200:
            text = (
                f"Predicted: {data['predicted_class']}\n"
                f"Confidence: {data['confidence']:.2%}"
            )
            result_label.config(text=text)
        else:
            result_label.config(text=f"Error: {data.get('detail', 'Unknown error')}")
    except Exception as e:
        result_label.config(text=f"Error: {e}")


# --- GUI setup ---
root = tk.Tk()
root.title("Disease Predictor")
root.geometry("400x450")

title_label = tk.Label(root, text="Ultrasound Disease Predictor", font=("Arial", 16))
title_label.pack(pady=10)

select_btn = tk.Button(root, text="Select Image & Predict", command=select_and_predict, font=("Arial", 12))
select_btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="No prediction yet.", font=("Arial", 14), justify="center")
result_label.pack(pady=10)

root.mainloop()
