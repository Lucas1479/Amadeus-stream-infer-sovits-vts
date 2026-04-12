from PIL import Image
import os

path = r"F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\render\assets\images\normal\loop\kurisu_normal_loop0001.png"
img = Image.open(path)
print("size:", img.size)
print("width:", img.size[0], "height:", img.size[1])
