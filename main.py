import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
from scipy import ndimage
import math

# --- Custom Slider Widget ---
class CanvasSlider(tk.Canvas):
    def __init__(self, parent, length=120, min_val=0, max_val=1, init_val=0, command=None, **kwargs):
        super().__init__(parent, width=30, height=length+20, bg=parent['bg'], highlightthickness=0, **kwargs)
        self.length = length
        self.min_val = min_val
        self.max_val = max_val
        self.command = command
        # line coordinates
        self.pad = 10
        self.x = 15
        self.y0 = self.pad
        self.y1 = self.pad + self.length
        # initial position
        self.val = init_val
        self._draw_line()
        self._draw_handle()
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._drag)

    def _draw_line(self):
        self.create_line(self.x, self.y0, self.x, self.y1, fill="#888", width=2)

    def _draw_handle(self):
        y = self._value_to_y(self.val)
        r = 6
        # remove previous
        if hasattr(self, 'handle'): self.delete(self.handle)
        self.handle = self.create_oval(self.x-r, y-r, self.x+r, y+r, fill="#444", outline="")

    def _value_to_y(self, val):
        frac = (val - self.min_val) / (self.max_val - self.min_val)
        return self.y1 - frac * self.length

    def _y_to_value(self, y):
        frac = (self.y1 - y) / self.length
        return max(self.min_val, min(self.max_val, self.min_val + frac * (self.max_val - self.min_val)))

    def _click(self, event):
        self._update_val(event.y)

    def _drag(self, event):
        self._update_val(event.y)

    def _update_val(self, y):
        self.val = self._y_to_value(y)
        self._draw_handle()
        if self.command:
            self.command(self.val)

    def get(self):
        return self.val

# --- Normal map utilities ---
def smooth_gaussian(im: np.ndarray, sigma) -> np.ndarray:
    if sigma == 0:
        return im
    kernel_x = np.arange(-3*sigma, 3*sigma+1).astype(float)
    kernel_x = np.exp(-(kernel_x**2) / (2*sigma**2))
    im_smooth = ndimage.convolve(im.astype(float), kernel_x[np.newaxis])
    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)
    return im_smooth


def sobel(im_smooth: np.ndarray):
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gx = ndimage.convolve(im_smooth.astype(float), kernel)
    gy = ndimage.convolve(im_smooth.astype(float), kernel.T)
    return gx, gy


def compute_normal_map(gx: np.ndarray, gy: np.ndarray, intensity=1.0) -> np.ndarray:
    max_val = max(gx.max(), gy.max(), 1e-6)
    nm = np.zeros((gx.shape[0], gx.shape[1], 3), dtype=float)
    nm[...,0] = gx / max_val
    nm[...,1] = gy / max_val
    nm[...,2] = 1.0 / (1.0/intensity)
    norm = np.linalg.norm(nm, axis=2, keepdims=True)
    nm /= norm
    nm = nm * 0.5 + 0.5
    return (nm * 255).astype(np.uint8)

# --- GUI Application ---
class CropApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Cropper + Normal Map")

        # Frames for layout
        self.frame_left = tk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT)
        self.frame_preview = tk.Frame(self.root)
        self.frame_preview.pack(side=tk.LEFT, padx=5)
        self.frame_controls = tk.Frame(self.root)
        self.frame_controls.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # Cropping canvas
        self.canvas = tk.Canvas(self.frame_left, cursor="cross")
        self.canvas.pack()

                # Preview canvases (stacked, each 2:1 aspect)
        self.canvas_crop = tk.Canvas(self.frame_preview, width=512, height=256)
        self.canvas_crop.pack(pady=5)
        self.canvas_normal = tk.Canvas(self.frame_preview, width=512, height=256)
        self.canvas_normal.pack(pady=5)

        # Controls
        tk.Button(self.frame_controls, text="Open", width=12, command=self.open_image).pack(pady=4)
        tk.Button(self.frame_controls, text="Save", width=12, command=self.save_images).pack(pady=4)

        # Normal map settings
        tk.Label(self.frame_controls, text="Normal Map Settings", font=(None,10,'bold')).pack(pady=(20,0))
        slider_frame = tk.Frame(self.frame_controls)
        slider_frame.pack(pady=5)
        tk.Label(slider_frame, text="Blur").grid(row=0, column=0)
        tk.Label(slider_frame, text="Intensity").grid(row=0, column=1)
        self.slider_sigma = CanvasSlider(
            slider_frame, min_val=0, max_val=10, init_val=0, length=120,
            command=lambda v: self.update_previews()
        )
        self.slider_sigma.grid(row=1, column=0, padx=5)
        self.slider_int = CanvasSlider(
            slider_frame, min_val=0.1, max_val=5, init_val=1, length=120,
            command=lambda v: self.update_previews()
        )
        self.slider_int.grid(row=1, column=1, padx=5)
        # Invert normal as part of normal settings
        tk.Button(self.frame_controls, text="Invert Normal", width=12, command=self.invert_normal).pack(pady=(5,10))

        # Actions: transformations
        tk.Label(self.frame_controls, text="Image Transformations", font=(None,10,'bold')).pack(pady=(20,0))
        tk.Button(self.frame_controls, text="Rotate 90°", width=12, command=self.rotate_preview).pack(pady=2)
        tk.Button(self.frame_controls, text="Flip Image", width=12, command=self.flip_preview).pack(pady=2)

                # Help and Credits buttons at bottom right
        tk.Button(self.frame_controls, text="Help", width=12, command=self.show_help).pack(side=tk.BOTTOM, pady=(5,2))
        tk.Button(self.frame_controls, text="Credits", width=12, command=self.show_credits).pack(side=tk.BOTTOM, pady=(2,10))

        # State
        self.img = None
        self.scale = 1.0
        self.handles = []
        self.lines = []
        self.labels = []
        self.drag_data = {"handle": None, "x": 0, "y": 0}
        self.cropped = None
        self.normal = None

        # Bind handle events
        self.canvas.tag_bind("handle", "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind("handle", "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind("handle", "<ButtonRelease-1>", self.on_release)

        self.root.mainloop()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not path:
            return
        from PIL import Image
        self.img = Image.open(path).convert("RGB")
        w, h = self.img.size
        max_w, max_h = 800,600
        self.scale = min(max_w/w, max_h/h, 1.0)
        disp = self.img.resize((int(w*self.scale), int(h*self.scale)), Image.BICUBIC)
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)
        self.canvas.delete("all")
        self.canvas.create_image(0,0,anchor=tk.NW, image=self.tkimg)
        self.init_handles()

    def init_handles(self):
        for item in self.handles + self.lines + self.labels:
            self.canvas.delete(item)
        self.handles.clear(); self.lines.clear(); self.labels.clear()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        m = min(w,h)//4
        coords = [(w//2-m, h//2-m), (w//2+m, h//2-m), (w//2+m, h//2+m), (w//2-m, h//2+m)]
        for x, y in coords:
            hndl = self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="red", tags="handle")
            self.handles.append(hndl)
        self.update_lines()

    def update_lines(self):
        for item in self.lines + self.labels:
            self.canvas.delete(item)
        self.lines.clear(); self.labels.clear()
        pts = [self._center(h) for h in self.handles]
        for i in range(4):
            x1, y1 = pts[i]; x2, y2 = pts[(i+1)%4]
            ln = self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
            self.lines.append(ln)

    def _center(self, h):
        x0, y0, x1, y1 = self.canvas.coords(h)
        return ((x0+x1)/2, (y0+y1)/2)

    def on_press(self, e):
        h = self.canvas.find_closest(e.x, e.y)[0]
        if h in self.handles:
            self.drag_data = {"handle": h, "x": e.x, "y": e.y}

    def on_drag(self, e):
        h = self.drag_data.get("handle")
        if h:
            dx, dy = e.x - self.drag_data['x'], e.y - self.drag_data['y']
            self.canvas.move(h, dx, dy)
            self.drag_data.update({"x": e.x, "y": e.y})
            self.update_lines()

    def on_release(self, e):
        self.update_previews()

    def update_previews(self, *args):
        if not self.img:
            return
        disp_pts = [self._center(h) for h in self.handles]
        orig_pts = [(x/self.scale, y/self.scale) for x, y in disp_pts]
        lens = [math.hypot(orig_pts[i][0]-orig_pts[(i+1)%4][0], orig_pts[i][1]-orig_pts[(i+1)%4][1]) for i in range(4)]
        mi = lens.index(max(lens))
        ordered = [orig_pts[(mi+i)%4] for i in range(4)]
        quad = [c for p in ordered for c in p]
        tr = self.img.transform((256, 512), Image.QUAD, quad, resample=Image.BICUBIC)
        tr = tr.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        self.cropped = tr
        gray = np.array(tr.convert('L'), dtype=float)
        smooth = smooth_gaussian(gray, self.slider_sigma.get())
        gx, gy = sobel(smooth)
        nm_arr = compute_normal_map(gx, gy, self.slider_int.get())
        self.normal = Image.fromarray(nm_arr)
        self.display_previews()

    def rotate_preview(self):
        if self.cropped:
            self.cropped = self.cropped.rotate(-90, expand=True)
            gray = np.array(self.cropped.convert('L'), dtype=float)
            smooth = smooth_gaussian(gray, self.slider_sigma.get())
            gx, gy = sobel(smooth)
            nm_arr = compute_normal_map(gx, gy, self.slider_int.get())
            self.normal = Image.fromarray(nm_arr)
            self.display_previews()

    def flip_preview(self):
        if self.cropped:
            self.cropped = self.cropped.transpose(Image.FLIP_LEFT_RIGHT)
            gray = np.array(self.cropped.convert('L'), dtype=float)
            smooth = smooth_gaussian(gray, self.slider_sigma.get())
            gx, gy = sobel(smooth)
            nm_arr = compute_normal_map(gx, gy, self.slider_int.get())
            self.normal = Image.fromarray(nm_arr)
            self.display_previews()

    def invert_normal(self):
        if self.normal:
            r, g, b = self.normal.split()
            g = ImageOps.invert(g)
            self.normal = Image.merge('RGB', (r, g, b))
            self.display_previews()

    def display_previews(self):
        # Show stacked crop and normal map on respective canvases
        disp_crop = self.cropped.resize((512, 256), Image.BICUBIC)
        disp_nm = self.normal.resize((512, 256), Image.BICUBIC)
        self.tk_crop = ImageTk.PhotoImage(disp_crop)
        self.canvas_crop.delete("all")
        self.canvas_crop.create_image(0, 0, anchor=tk.NW, image=self.tk_crop)
        self.tk_nm = ImageTk.PhotoImage(disp_nm)
        self.canvas_normal.delete("all")
        self.canvas_normal.create_image(0, 0, anchor=tk.NW, image=self.tk_nm)

    def show_help(self):
        messagebox.showinfo(
            "Help\n",
            "Instructions:\n"
            "1. Click 'Open' to load an image.\n"
            "2. Drag the red corner handles to adjust the crop quadrilateral.\n"
            "3. Adjust 'Blur' and 'Intensity' sliders to tweak the normal map.\n"
            "4. Click 'Save' to export the cropped image and normal map (_N suffix).\n"
            "Issues/Troubleshooting:\n"
            "- Use 'Rotate 90°' and 'Flip Image' under Image Transformations if orientation is incorrect.\n"
            "- Click 'Invert Normal' to flip the normal map's green channel if lighting appears reversed.\n"
        )

    def show_credits(self):
        messagebox.showinfo(
            "Credits\n",
            "Author: ModernLancer\n"
            "This app incorporates code from:\n"
            "Mehdi-Antoine's NormalMapGenerator\n"
            "https://github.com/Mehdi-Antoine/NormalMapGenerator/tree/master"
        )

    def save_images(self):
        if self.cropped:
            path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
            if path:
                self.cropped.save(path)
                base, ext = path.rsplit('.', 1)
                nm_path = f"{base}_N.{ext}"
                self.normal.save(nm_path)

if __name__ == "__main__":
    CropApp()
