import os
import cv2
import pytesseract
from PIL import Image
import json
import base64
import sys

# 检测矩形框的函数保持不变
import paddledet as pdet
from paddledet.apis import infer, init_model, get_test_images

def detect_rectangles(image_path):
    # 初始化模型
    model = init_model('path/to/config', 'path/to/model_weights')

    # 载入图像
    image = pdet.data.transforms.build_transforms(image_path)

    # 执行推理
    results = infer(model, image)

    # 解析检测结果，假设我们的目标类别标签是"rectangle"
    rectangles = []
    for bbox in results['boxes']:
        if bbox[0] == "rectangle":  # 假设模型输出的类别索引或名称是"rectangle"
            x1, y1, x2, y2, score = bbox[1:]
            if score > 0.5:  # 只包括置信度高于0.5的结果
                # 转换为(x, y, width, height)
                rectangles.append((x1, y1, x2 - x1, y2 - y1))
    return rectangles


# 文本检测函数保持不变
def detect_text(image_path):
    img = Image.open(image_path)
    text_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    text_boxes = []
    for i in range(len(text_data['text'])):
        if int(text_data['conf'][i]) > 60:
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_boxes.append((text_data['text'][i], x, y, w, h))
    return text_boxes

# 匹配矩形和文本的函数保持不变
def match_rectangles_and_text(rectangles, texts):
    annotations = []
    for rect in rectangles:
        rx, ry, rw, rh = rect
        matched = False
        for text, tx, ty, tw, th in texts:
            # 考虑文本左下角(tx, ty + th)与矩形左上角(rx, ry)的关系
            # 假设允许一定的容错范围（例如5像素以内）
            if abs(rx - tx) <= 5 and abs(ry - (ty + th)) <= 5:
                annotations.append({'shape': rect, 'label': text})
                matched = True
                break
        if not matched:
            # 如果没有找到匹配的文本，我们可以添加一个没有标签的矩形框
            annotations.append({'shape': rect, 'label': 'unknown'})
    return annotations


# 图像转换为Base64编码的函数保持不变
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# 写入JSON的函数保持不变
def write_labelme_json(annotations, image_path, output_file):
    image_data = image_to_base64(image_path)
    data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.split("/")[-1],
        "imageData": image_data
    }
    for annotation in annotations:
        shape, label = annotation['shape'], annotation['label']
        x, y, w, h = shape
        data['shapes'].append({
            "label": label,
            "points": [[x, y], [x, y + h], [x + w, y + h], [x + w, y]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# 处理单个图像或文件夹的逻辑
def process_images(input_path):
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_path, filename)
                process_single_image(image_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_single_image(input_path)
    else:
        print("Invalid input path. Please provide a path to an image file or a directory containing image files.")

# 处理单个图像的逻辑
def process_single_image(image_path):
    print(f"Processing {image_path}")
    rectangles = detect_rectangles(image_path)
    texts = detect_text(image_path)
    annotations = match_rectangles_and_text(rectangles, texts)
    output_file = os.path.splitext(image_path)[0] + '_labelme.json'
    write_labelme_json(annotations, image_path, output_file)

# 主程序入口
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_image_or_directory>")
    else:
        process_images(sys.argv[1])
