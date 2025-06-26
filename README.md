```markdown
# 🧠 부비동염 판별 시스템 (Sinusitis Detection UI)

부비동 X-ray 이미지를 업로드하면, AI 모델이 8개 클래스 중 어떤 상태인지 자동 판별하고 Grad-CAM으로 시각적으로 설명해주는 웹 기반 진단 보조 시스템입니다.

---

## 📁 폴더 구조

```

SINUS\_APP/
├── app.py                          # Flask 메인 실행 서버
├── model/
│   └── LSG\_model.h5               # 학습된 분류 모델
├── static/
│   └── logo.png                   # 웹 UI용 이미지
├── templates/
│   └── index.html                 # 메인 웹 페이지
├── utils/
│   ├── gradcam.py                 # Grad-CAM
│   ├── guided\_backprop.py        # Guided Backprop
│   ├── guided\_gradcam.py         # Guided Grad-CAM
│   └── preprocess.py             # 이미지 전처리
└── README.md                      # 설명 문서

````

---

## 💻 실행 방법

### 1. 패키지 설치
```bash
pip install flask tensorflow opencv-python matplotlib
````

### 2. 서버 실행

```bash
python app.py
```

### 3. 웹 페이지 접속

브라우저에서 [http://127.0.0.1:5000](http://127.0.0.1:5000) 입력

---

## 🧩 핵심 코드 요약

### 🔸 `app.py` - 서버 진입점

```python
from flask import Flask, render_template, request, jsonify
from utils.preprocess import preprocess_image
from utils.gradcam import generate_gradcam
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/LSG_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = preprocess_image(file)
    pred = model.predict(np.expand_dims(image, axis=0))[0]
    gradcam_path = generate_gradcam(model, image)
    return jsonify({'prediction': pred.tolist(), 'gradcam_path': gradcam_path})

if __name__ == '__main__':
    app.run(debug=True)
```

---

### 🔸 `templates/index.html` - 사용자 인터페이스

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>부비동염 판별</title>
</head>
<body>
  <h2>🔍 부비동염 진단 보조 시스템</h2>
  <form id="upload-form" enctype="multipart/form-data">
    <input type="file" name="file" id="file-input" accept="image/*" required>
    <button type="submit">진단하기</button>
  </form>
  <div id="result"></div>

  <script>
    const form = document.getElementById('upload-form');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('file-input');
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      document.getElementById('result').innerHTML =
        '<h3>예측 결과:</h3><pre>' + JSON.stringify(result.prediction, null, 2) + '</pre>' +
        '<img src="' + result.gradcam_path + '" alt="Grad-CAM 시각화" />';
    };
  </script>
</body>
</html>
```

---

### 🔸 `utils/preprocess.py` - 이미지 전처리

```python
import cv2
import numpy as np

def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (96, 96)) / 255.0
    return img.astype(np.float32)
```

---

### 🔸 `utils/gradcam.py` - Grad-CAM 생성

```python
import tensorflow as tf
import numpy as np
import cv2
import uuid
import os

def generate_gradcam(model, img, layer_name='conv_layer'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img, axis=0))
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (96, 96))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_rgb = np.uint8(255 * img)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    path = f'static/gradcam_{uuid.uuid4().hex}.png'
    cv2.imwrite(path, superimposed_img)
    return path
```

---

## 📸 웹 UI 예시

> 예시 이미지를 다음과 같이 넣을 수 있습니다.

```markdown
![예측 결과](static/example_result.png)
```

---

## 📚 참고

* 모델: EfficientNetB3 기반 8-class 분류기
* 시각화: Grad-CAM, Guided Backpropagation
* 프레임워크: Flask, HTML, OpenCV, TensorFlow
* 적용 분야: 부비동염 진단 보조

---