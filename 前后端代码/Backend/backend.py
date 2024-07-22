import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from flask import Flask, request, send_file
from fpdf import FPDF


from reportlab.platypus import Image as ImageOne



from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import  Paragraph, SimpleDocTemplate, Spacer,Table
from reportlab.lib.styles import getSampleStyleSheet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset

from PIL import Image
import numpy as np
from PIL import Image

# 读取图片
from Backend.all_color_modified import generate
from Backend.one_test import predict

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'edit_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER_2= '../frontend/src/assets/'
app.config['UPLOAD_FOLDER_2'] = UPLOAD_FOLDER_2

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # 获取上传的文件
    image_file = request.files['image']

    # 保存上传的图片为"a.jpg"
    image_file.save('data/test.png')


    # 返回成功消息
    return jsonify({'message': 'Image saved successfully as a.jpg.'}), 200


@app.route('/choose_model', methods=['POST'])
def choose_model():
    # 获取选择的模型类型
    selected_model = request.json.get('selectedModel')

    # 在这里可以进一步处理选择的模型类型，比如加载对应的模型

    generate(selected_model, 'data/test.png')

    print(selected_model)
    result_image_path = 'hahaha'

    # 返回消息和图片路径
    level = predict()  # 这里假设返回的等级是5，你可以根据实际逻辑设置

    # 返回消息、图片路径和等级
    return jsonify(
        {
            'message': result_image_path,
            'level': level
        }), 200


@app.route('/edit_image', methods=['POST'])
def edit_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)

    file.save(os.path.join(app.config['UPLOAD_FOLDER_2'], filename))
    time.sleep(1)
    file.stream.seek(0)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print("get there")
    return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 200


@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    name = data.get('name')
    age = data.get('age')

    diagnosis = data.get('diagnosis')
    grade = data.get('grade')


    pdf_output = 'report.pdf'
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Add title
    title = Paragraph("Medical Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Add user information
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Current Time: {current_time}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Name: {name}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Age: {age}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add first image section
    elements.append(Paragraph("Original Image", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(ImageOne("data/test.png", width=150, height=150))
    elements.append(Spacer(1, 12))

    # Add second image section
    elements.append(Paragraph("Model Result", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(ImageOne("result_images/result.png", width=150, height=150))
    elements.append(Spacer(1, 12))


    # Add four images in a row
    image_dir = 'result_images'
    images = [os.path.join(image_dir, image) for image in os.listdir(image_dir) if image.endswith('.png')]
    images = images[1:5]  # 仅取前四张图片

    if images:
        image_elements = [ImageOne(image_path, width=90, height=90) for image_path in images]
        table = Table([image_elements])
        elements.append(table)
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Prediction of lesion grade given by model: {grade}", styles['Normal']))
    elements.append(Spacer(1, 12))


    # Add modified result image
    elements.append(Paragraph("Modified Image", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(ImageOne("edit_data/edited_image.png", width=150, height=150))
    elements.append(Spacer(1, 12))

    # Add diagnosis
    elements.append(Paragraph(f"Diagnosis: {diagnosis}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)

    return send_file(pdf_output, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
