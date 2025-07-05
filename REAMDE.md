# Image Cropper + Normal Map

A lightweight Tkinter application for cropping images via an adjustable quadrilateral and generating normal maps from the cropped region.

## Architecture

* **`main.py`**: Entry point that launches the `CropApp` class.
* **`CanvasSlider`**: Custom Tkinter canvas widget for vertical sliders with draggable handles.
* **Image Cropping**: Uses four draggable red corner handles to define a quadrilateral, then applies a perspective transform (`PIL.Image.transform`) to a fixed aspect output.
* **Normal Map Generation**:

  * Converts the cropped image to grayscale.
  * Applies a Gaussian-like smoothing (`smooth_gaussian` using SciPy convolutions).
  * Computes gradients via a Sobel filter (`sobel`).
  * Packs X/Y gradients and an intensity-based Z component into RGB, normalizes, and remaps to 0–255.
* **UI Layout**:

  * **Left pane**: Original image with corner handles.
  * **Center pane**: Two stacked 512×256 canvases showing the cropped image and its normal map.
  * **Right pane**: Controls for opening/saving, normal-map sliders (blur & intensity), invert-normal, and image transformations (rotate/flip), plus Help & Credits buttons.

## Usage

1. **Open** an image file (`PNG`, `JPG`, `BMP`, etc.).
2. **Drag** the red dots to adjust the crop area.
3. **Adjust** **Blur** and **Intensity** using the sliders.
4. **Invert Normal** to flip the green channel if needed.
5. **Rotate 90°** or **Flip Image** for orientation fixes.
6. **Save** to export:

   * Cropped image (`filename.png`)
   * Normal map (`filename_N.png`)

## Packaging into an EXE

Build a standalone Windows executable with PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --strip main.py
```

* `--onefile`: Produces a single `main.exe` standalone file.
* `--windowed`: Suppresses the console window.
* `--strip`: Removes symbol tables to reduce size.

The resulting EXE in `dist/` has no outbound connections and minimal dependencies, suitable for distribution without antivirus flags.

## Credits

* **Author**: ModernLancer
* **Normal Map Algorithm**: Mezhi-Antoine's [NormalMapGenerator](https://github.com/Mehdi-Antoine/NormalMapGenerator/tree/master)

---

*For more details, see the source code and docstrings in `main.py`.*
