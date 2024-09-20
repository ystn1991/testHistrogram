import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพ
image = cv2.imread('Source/tonkla.jpg')


# แปลงภาพจาก BGR เป็น RGB (เพื่อแสดงใน Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# สร้างหน้าต่างย่อยสำหรับแสดงผล
fig, axs = plt.subplots(4, 2, figsize=(14, 10))

# แสดงภาพต้นฉบับ
axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Original Image (RGB)")
axs[0, 0].axis('off')

# แสดง Histogram รวมของทุกสี
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    axs[0, 1].plot(hist, color=color)
    axs[0, 1].set_title('Combined Histogram (RGB)')

# แสดงภาพและ Histogram แยกตามสี
color_channels = ['Red', 'Green', 'Blue']
for i, color in enumerate(colors):
    # แสดงภาพตามช่องสี
    blank_image = np.zeros_like(image_rgb)
    blank_image[:, :, i] = image_rgb[:, :, i]
    axs[i+1, 0].imshow(blank_image)
    axs[i+1, 0].set_title(f"{color_channels[i]} Channel")
    axs[i+1, 0].axis('off')

    # แสดง Histogram
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    axs[i+1, 1].plot(hist, color=color)
    axs[i+1, 1].set_title(f"{color_channels[i]} Histogram")


plt.tight_layout()
plt.show()
