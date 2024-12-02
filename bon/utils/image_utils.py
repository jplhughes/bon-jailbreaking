import base64
import io
from pathlib import Path

import cv2
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vertexai.generative_models import Part


def get_default_image(text: str, height: int = 320, width: int = 320):
    image = None
    for i in range(20):
        init_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        kwargs = {
            "image": init_image,
            "text": text,
            "position": (40, 40),
            "font_scale": 1.2 - i * 0.05,
            "thickness": 2,
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "color": (255, 255, 255),
        }

        image = add_text_to_image(**kwargs)
        if image is not None:
            break
    if image is None:
        raise ValueError("Failed to add text to image")
    return image


def find_working_fonts(number_to_find: int = 200):
    working_fonts = []
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    for font in range(number_to_find):
        try:
            cv2.putText(image, "Hello", (0, 0), font, 1, (255, 255, 255), 1)
            working_fonts.append(font)
        except Exception:
            continue

    return working_fonts


def add_text_to_image(
    image,
    text,
    position=(0, 0),
    font_scale: float = 1,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray | None:
    """
    Adds text to an image at a specified position with given font properties.

    Parameters:
    image (numpy.ndarray): The image to which text will be added. (height, width, 3)
    text (str): The text to add to the image.
    position (tuple): The (x, y) position where the text will be placed.
    font_scale (float): The scale factor for the font size.
    thickness (int): The thickness of the text strokes.
    font (int): The font type to be used.
    color (tuple): The color of the text in (B, G, R) format.

    Returns:
    numpy.ndarray: The image with the added text.
    """

    # assert font in VALID_FONTS, "Invalid font"
    assert all(0 <= c <= 255 for c in color), "Invalid color"
    assert 1 <= thickness <= 6, "Invalid thickness"
    assert 0.05 <= font_scale <= 2, "Invalid font scale"
    assert 0 <= position[0] <= image.shape[1] // 2, "Invalid x position"
    assert 0 <= position[1] <= image.shape[0] // 2, "Invalid y position"

    max_width = (image.shape[1] - position[0]) * 0.8
    words = text.split()
    wrapped_text = ""
    line = ""

    for word in words:
        test_line = line + word + " "
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width:
            wrapped_text += line + "\n"

            (w1, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if w1 > (image.shape[1] - position[0]):
                return None

            line = word + " "
        else:
            line = test_line
    wrapped_text += line

    # Check if text height fits
    y0, dy = position[1], int(h * 1.5)
    for i, line in enumerate(wrapped_text.split("\n")):
        y = y0 + i * dy
        if y + h > image.shape[0]:
            return None
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def image_to_base64(image: np.ndarray | Path | str) -> str:
    if isinstance(image, np.ndarray):
        img = Image.fromarray((image * 255).astype("uint8"))
    elif isinstance(image, Path) or isinstance(image, str):
        img = Image.open(image)
    else:
        raise ValueError(f"Invalid image type {type(image)}")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def save_image_from_array(image: np.ndarray, filename: str):
    # assert range of image is 0-1
    if np.max(image) <= 1:
        img = Image.fromarray((image * 255).astype("uint8"))
    else:
        img = Image.fromarray(image.astype("uint8"))
    img.save(filename)


def load_image_from_file(filename: str) -> np.ndarray:
    img = Image.open(filename)
    return np.array(img) / 255.0


def display_image_without_frame(image: np.ndarray, height: int = 4, width: int = 4):
    # Create a figure without a frame
    fig = plt.figure(frameon=False)
    aspect_ratio = image.shape[1] / image.shape[0]
    if aspect_ratio > 1:
        fig.set_size_inches(height * aspect_ratio, height)
    else:
        fig.set_size_inches(width, width / aspect_ratio)

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(image, aspect="auto")
    plt.show()


def get_image_file_type(image_file: str) -> str:
    if Path(image_file).suffix.lower() in [".jpg", ".jpeg"]:
        return "jpeg"
    elif Path(image_file).suffix.lower() in [".png"]:
        return "png"
    else:
        raise ValueError(f"Image path {image_file} is not a valid media_type!")


def prepare_gemini_image(image_file: str, use_vertexai: bool = False) -> Part | genai.types.file_types.File:
    if use_vertexai:
        encoded_image = image_to_base64(image_file)
        image_type = get_image_file_type(image_file)
        return Part.from_data(data=encoded_image, mime_type=f"image/{image_type}")
    else:
        image = genai.upload_file(image_file)
        return image


def basic_text_image(text: str, path: str):
    image_with_text = add_text_to_image(
        image=np.zeros((400, 400, 3), dtype=np.uint8),
        font_scale=0.3,
        thickness=1,
        text=text,
        position=(10, 20),
    )
    save_image_from_array(image_with_text, path)
    return image_with_text
