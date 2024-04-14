from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detailed_sketch(image_path, blur_value=11, blockSize=9, C=2):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray_image, blur_value)
    sketch_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
    return sketch_image

def sketchify(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred_image)
    sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)
    return sketch_image

def cartoonify(image):
    color = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
    return cartoon

def apply_pop_art_effect(image):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    simplified_image = res.reshape((image.shape))
    hsv = cv2.cvtColor(simplified_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)
    pop_art_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return pop_art_image

@app.route('/', methods=['GET', 'POST'])
def home():
    image_url = None
    if request.method == 'POST':
        effect = request.form.get('effect')
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            original_image = cv2.imread(input_path)
            
            if effect == '1':
                result_image = detailed_sketch(input_path, blur_value=11, blockSize=9, C=2)
            elif effect == '2':
                result_image = cartoonify(original_image)
            elif effect == '3':
                result_image = apply_pop_art_effect(original_image)
            elif effect == '4':
                result_image = sketchify(original_image)
            else:
                return "Invalid choice", 400
            
            output_filename = 'output_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, result_image)
            
            image_url = url_for('static', filename='uploads/' + output_filename)

    return render_template('index.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
