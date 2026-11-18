"""Small desktop window: choose a photo, click "Clean moiré", save the result."""
import base64
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from moire.image_io import load_image, save_image
from moire.pipeline import moire_filter_and_screen_clean, process_image

PREVIEW_SIZE = (460, 460)  # max preview width, height (px)
IMAGE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]


def to_photo_image(image_bgr, max_size=PREVIEW_SIZE):
    """Convert a BGR image to a downscaled tkinter PhotoImage for preview."""
    h, w = image_bgr.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1.0)
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, png = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not create preview.")
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()))


class App:
    def __init__(self, root, initial_dir=None):
        self.root = root
        self.initial_dir = initial_dir
        self.image = None
        self.result = None
        self.source_path = None
        self.results = queue.Queue()
        self.photos = {}  # keep references so tkinter does not drop the previews

        root.title("Moiré cleaner")

        bar = ttk.Frame(root, padding=10)
        bar.pack(fill="x")
        self.choose_btn = ttk.Button(bar, text="Choose image…", command=self.choose)
        self.choose_btn.pack(side="left")
        self.clean_btn = ttk.Button(
            bar, text="Clean moiré", command=self.clean, state="disabled"
        )
        self.clean_btn.pack(side="left", padx=8)
        self.save_btn = ttk.Button(
            bar, text="Save result…", command=self.save, state="disabled"
        )
        self.save_btn.pack(side="left")

        views = ttk.Frame(root, padding=(10, 0))
        views.pack(fill="both", expand=True)
        self.labels = {}
        for column, name in enumerate(("Original", "Cleaned")):
            frame = ttk.LabelFrame(views, text=name, padding=5)
            frame.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")
            label = ttk.Label(frame, anchor="center")
            label.pack()
            self.labels[name] = label
            views.columnconfigure(column, weight=1)

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=10)
        self.status = tk.StringVar(value="Choose an image to get started.")
        ttk.Label(root, textvariable=self.status, padding=10).pack(fill="x")

    def show(self, name, image):
        photo = to_photo_image(image)
        self.photos[name] = photo
        self.labels[name].configure(image=photo)

    def choose(self):
        path = filedialog.askopenfilename(
            title="Choose an image to clean",
            initialdir=self.initial_dir,
            filetypes=IMAGE_TYPES,
        )
        if not path:
            return
        try:
            image = load_image(path)
        except FileNotFoundError:
            messagebox.showerror("Moiré cleaner", f"Could not open this file as an image:\n{path}")
            return
        self.image, self.result, self.source_path = image, None, Path(path)
        self.show("Original", image)
        self.labels["Cleaned"].configure(image="")
        self.photos.pop("Cleaned", None)
        self.save_btn.configure(state="disabled")
        self.clean_btn.configure(state="normal")
        h, w = image.shape[:2]
        self.status.set(f"{self.source_path.name} ({w} × {h}). Click “Clean moiré”.")

    def clean(self):
        self.clean_btn.configure(state="disabled")
        self.choose_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.status.set("Cleaning… large photos can take a while.")
        self.progress.start(12)
        threading.Thread(target=self._work, args=(self.image,), daemon=True).start()
        self.root.after(100, self._poll)

    def _work(self, image):
        try:
            self.results.put(("ok", process_image(image, moire_filter_and_screen_clean)))
        except Exception as error:  # report any failure in the window
            self.results.put(("error", error))

    def _poll(self):
        try:
            kind, value = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        self.progress.stop()
        self.choose_btn.configure(state="normal")
        self.clean_btn.configure(state="normal")
        if kind == "error":
            self.status.set("Something went wrong.")
            messagebox.showerror("Moiré cleaner", str(value))
            return
        self.result = value
        self.show("Cleaned", value)
        self.save_btn.configure(state="normal")
        self.status.set("Done. Click “Save result…” to keep the cleaned image.")

    def save(self):
        default = f"{self.source_path.stem}_cleaned.png"
        path = filedialog.asksaveasfilename(
            title="Save cleaned image",
            initialdir=self.source_path.parent,
            initialfile=default,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if path:
            save_image(path, self.result)
            self.status.set(f"Saved: {path}")


def run(initial_dir=None):
    root = tk.Tk()
    App(root, initial_dir)
    root.mainloop()
