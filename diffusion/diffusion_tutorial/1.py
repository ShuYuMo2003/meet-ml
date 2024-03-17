import torch
from torchvision import transforms
from PIL import Image

# 创建一个Resize实例，并指定目标尺寸
resize_size = (100, 256)
resizer = transforms.Resize(size=resize_size)

# 读取图像
image = Image.open("a.png")

# 进行尺寸调整
print(transforms.ToTensor()(image).shape)
resized_image = resizer(image)
print(transforms.ToTensor()(resized_image).shape)
resized_image_center = transforms.CenterCrop(100)(resized_image)
print(transforms.ToTensor()(resized_image_center).shape)

# 显示调整后的图像

resized_image.save("saved_image.png")
resized_image_center.save("resized_image_center.png")