import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random

# 配置参数
image_folder = Path('/media/yf/269E116F9E1138AF/巡检结果/Baodi2kanchayuan_上行_20240516154904913/巡检输出/原始数据/图像')
output_image_folder = Path('/home/yf/Documents/data/fake/images')
output_ann_folder = Path('/home/yf/Documents/data/fake/anns')
font_folder = Path('/path/to/your/font/folder')  # 字体文件夹路径
categories = [
    "横向裂缝", "纵向裂缝", "网状裂缝", "块状修补", "路面异常",
    "带状修补", "坑槽", "抛洒物", "标志缺损", "标线模糊", "护栏缺损",
    "中分带绿植缺死", "防眩板缺损", "指路标志", "信号灯", "枪机",
    "球机", "禁令标志", "指示标志", "警告标志", "辅助标志"
]

# 加载所有字体路径
fonts = [font_folder / f for f in os.listdir(font_folder) if f.endswith('.ttf')]

# 初始化COCO格式的数据字典
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{'id': idx + 1, 'name': name} for idx, name in enumerate(categories)]
}
annotation_id = 1

# 读取并处理每张图像
for i, img_path in tqdm(enumerate(image_folder.glob('*.jpg'))):
    img = cv2.imread(str(img_path))
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    height, width, _ = img.shape

    # 图像信息
    image_info = {
        "file_name": img_path.name,
        "height": height,
        "width": width,
        "id": int(i + 1)  # 确保是标准Python int类型
    }
    coco_format['images'].append(image_info)

    # 随机生成矩形框
    num_boxes = np.random.randint(1, 11)
    for _ in range(num_boxes):
        margin = 10
        max_width = 300
        max_height = 300
        w = int(np.random.randint(20, min(max_width, width - 2 * margin)))
        h = int(np.random.randint(20, min(max_height, height - 2 * margin)))
        x = int(np.random.randint(margin, width - w - margin))
        y = int(np.random.randint(margin, height - h - margin))
        color = (int(np.random.randint(255)), int(np.random.randint(255)), int(np.random.randint(255)))
        category_id = int(np.random.choice(range(1, len(categories) + 1)))
        category_name = categories[category_id - 1]

        # 随机选择一个字体
        font_path = random.choice(fonts)
        font = ImageFont.truetype(str(font_path), 20)

        # 绘制矩形框
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=3)

        # 写文字，多次绘制以模拟加粗效果
        text_size = draw.textsize(category_name, font=font)
        text_x = x + (w - text_size[0]) // 2
        text_y = y - text_size[1] - 10
        for offset in [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]:
            draw.text((text_x + offset[0], text_y + offset[1]), category_name, fill="black", font=font)

        draw.rectangle([(text_x, text_y), (text_x + text_size[0], text_y + text_size[1])], outline=color, width=1)

        # 标注信息
        annotation = {
            "id": annotation_id,
            "image_id": int(i + 1),  # 确保是标准Python int类型
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }
        coco_format['annotations'].append(annotation)
        annotation_id += 1

    # 保存修改后的图像
    img_with_annotations = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_image_folder / img_path.name), img_with_annotations)

# 将所有numpy数据类型转换为标准Python数据类型
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

coco_format = json.loads(json.dumps(coco_format, default=convert_numpy))

# 保存COCO标注文件
with open(output_ann_folder / 'annotations.json', 'w') as f:
    json.dump(coco_format, f, indent=4)

print("Dataset creation complete!")
