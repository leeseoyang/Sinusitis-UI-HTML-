# 🔧 app.py - 수정된 통합 버전
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import io
import base64
from PIL import Image
from utils.preprocess import preprocess_and_correct
from utils.guided_gradcam import guided_gradcam

# Flask 앱 초기화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

# 모델 로딩 및 클래스 정의
model = tf.keras.models.load_model('model/LSG_model.h5', compile=False)
class_names = ['Normal', 'Left', 'Right', 'Both']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 🔹 이미지 로딩
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    # 🔹 전처리 + 좌우 반전 보정
    image_for_model, flip_flag, corrected_pil = preprocess_and_correct(image)

    # 🔹 예측
    preds = model.predict(np.expand_dims(image_for_model, axis=0))[0]
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds) * 100

    # 🔹 Grad-CAM
    _, guided_encoded = guided_gradcam(
        model, np.expand_dims(image_for_model, axis=0), pred_index, last_conv_layer_name='conv2d_2'
    )

    # 🔹 이미지 base64 인코딩
    img_byte_arr = io.BytesIO()
    corrected_pil.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return render_template('index.html',
        prediction=pred_class,
        confidence=confidence,
        probs=preds.tolist(),
        class_names=class_names,
        image_data=img_base64,
        guided_gradcam_img=guided_encoded,
        flip_flag=flip_flag
    )

if __name__ == '__main__':
    app.run(debug=True)
# Flask 앱 실행
# if __name__ == '__main__':