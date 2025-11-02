from __future__ import annotations

"""
Flask 웹앱: 부비동염 X-ray 이미지 분류
8클래스 모델 (ckpt_best.h5) + 4클래스 모델 (LSG_model.h5) 지원
"""
import json
import io
import base64
from typing import Dict, Optional, Any, TYPE_CHECKING
from flask import Flask, render_template, request

# TensorFlow 모델 타입 정의 (타입 체킹용)
if TYPE_CHECKING:
    try:
        from tensorflow.keras.models import Model  # type: ignore
        ModelType = Model  # type: ignore
    except ImportError:
        ModelType = Any
else:
    ModelType = Any

app = Flask(__name__)

# 기본 클래스 이름들
_default_class_names_8 = [
    'Normal',
    'Left-Mucosal',
    'Left-Air Fluid',
    'Left-Haziness', 
    'Right-Mucosal',
    'Right-Air Fluid',
    'Right-Haziness',
    'Both'
]

_default_class_names_4 = [
    'Normal',
    'Left-Sinusitis', 
    'Right-Sinusitis',
    'Bilateral-Sinusitis'
]

def _load_models_and_classes() -> tuple[Any, Any, list[str], list[str]]:
    """모델과 클래스 이름을 로딩하는 함수"""
    model_8class: Optional[Any] = None
    model_4class: Optional[Any] = None
    class_names_8 = _default_class_names_8.copy()
    class_names_4 = _default_class_names_4.copy()

    try:
        from tensorflow import keras  # type: ignore
        
        # 8클래스 모델 로딩 (ckpt_best.h5)
        try:
            model_8class = keras.models.load_model('model/ckpt_best.h5', compile=False)  # type: ignore
            print("✅ 8클래스 모델 (ckpt_best.h5) 로딩 성공")
        except Exception as e:
            print(f"❌ 8클래스 모델 로딩 실패: {e}")
            model_8class = None
        
        # 4클래스 모델 로딩 (LSG_model.h5)
        try:
            model_4class = keras.models.load_model('model/LSG_model.h5', compile=False)  # type: ignore
            print("✅ 4클래스 모델 (LSG_model.h5) 로딩 성공")
        except Exception as e:
            print(f"❌ 4클래스 모델 로딩 실패: {e}")
            model_4class = None
        
        # 메타 파일에서 클래스 이름 읽기 (선택적)
        try:
            with open('model/model_meta.json', 'r', encoding='utf-8') as meta_file:
                meta = json.load(meta_file)
            class_names_8 = meta.get('class_names_8', _default_class_names_8)
            class_names_4 = meta.get('class_names_4', _default_class_names_4)
        except (FileNotFoundError, json.JSONDecodeError):
            class_names_8 = _default_class_names_8
            class_names_4 = _default_class_names_4
            
    except Exception as e:
        print(f"❌ TensorFlow 로딩 실패: {e}")
        # TensorFlow/모델 로딩 실패 시 None 유지, 기본 클래스 사용
        model_8class = None
        model_4class = None
    
    return model_8class, model_4class, class_names_8, class_names_4  # type: ignore

# 앱 시작 시 한 번 시도 (실패하더라도 서버는 뜨고, 예측 시점에 안내)
model_8class: Any
model_4class: Any
class_names_8: list[str] 
class_names_4: list[str]

model_8class, model_4class, class_names_8, class_names_4 = _load_models_and_classes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 지연 임포트 (개발 환경에서 불필요한 임포트 오류 완화)
    from PIL import Image
    import numpy as np
    from utils.preprocess import preprocess_and_correct  # type: ignore

    # 모델 선택 (기본값: 8클래스)
    model_type = request.form.get('model_type', '8class')
    
    # 이미지 로딩/검증
    if 'image' not in request.files:
        return render_template('index.html', prediction=None, 
                               class_names_8=class_names_8, class_names_4=class_names_4,
                               probs=None, image_data=None, boxed_image_data=None, 
                               left_score=None, right_score=None, model_type=model_type)
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction=None,
                               class_names_8=class_names_8, class_names_4=class_names_4,
                               probs=None, image_data=None, boxed_image_data=None, 
                               left_score=None, right_score=None, model_type=model_type)

    try:
        image = Image.open(file.stream)
    except Exception:
        return render_template('index.html', prediction=None,
                               class_names_8=class_names_8, class_names_4=class_names_4,
                               probs=None, image_data=None, boxed_image_data=None, 
                               left_score=None, right_score=None, model_type=model_type)

    # 전처리 (모델 타입에 따라 채널 수 조정)
    # 첫 번째 에러에서 8클래스 모델이 1채널을 기대, 4클래스 모델이 3채널을 기대
    if model_type == '4class':
        channels = 3  # LSG 모델은 3채널
    else:
        channels = 1  # ckpt_best 모델은 1채널
    
    image_for_model, corrected_pil = preprocess_and_correct(image, channels=channels)  # type: ignore

    # 모델 선택 및 준비 확인
    if model_type == '4class':
        selected_model = model_4class
        selected_class_names = class_names_4
    else:
        selected_model = model_8class
        selected_class_names = class_names_8
    
    if selected_model is None:
        # 모델 미로딩/TF 미설치 상태에서는 입력/전처리만 표시
        img_byte_arr = io.BytesIO()
        corrected_pil.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return render_template('index.html', prediction=None,
                               class_names_8=class_names_8, class_names_4=class_names_4,
                               probs=None, image_data=img_base64, boxed_image_data=None, 
                               left_score=None, right_score=None, model_type=model_type,
                               error_msg=f'{model_type} 모델을 찾을 수 없습니다.')

    # 예측 (이미 (96, 96, 1) 형태로 전처리됨)
    preds = selected_model.predict(image_for_model[np.newaxis, ...], batch_size=1)[0]  # type: ignore
    pred_index = int(np.argmax(preds))  # type: ignore
    pred_class = selected_class_names[pred_index]
    confidence = float(np.max(preds) * 100.0)  # type: ignore

    # 박스 기반 자동 추적 오버레이 생성
    boxed_base64 = None
    side_scores = {"left": 0.0, "right": 0.0}
    try:
        import cv2
        from utils.roi import summarize_side_scores, draw_boxes_on_image

        side_scores: Dict[str, float] = summarize_side_scores(preds, selected_class_names)  # type: ignore
        bgr = cv2.cvtColor(np.array(corrected_pil.convert('L')), cv2.COLOR_GRAY2BGR)
        boxed_bgr = draw_boxes_on_image(bgr.copy(), side_scores, label=pred_class, conf=confidence)  # type: ignore

        buf2 = io.BytesIO()
        img_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)  # type: ignore
        Image.fromarray(img_rgb).save(buf2, format='PNG')  # type: ignore
        boxed_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    except Exception:
        boxed_base64 = None

    # 이미지 base64 인코딩
    img_byte_arr = io.BytesIO()
    corrected_pil.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return render_template('index.html',
        prediction=pred_class,
        confidence=confidence,
        probs=preds.tolist(),  # type: ignore
        class_names_8=class_names_8,
        class_names_4=class_names_4,
        selected_class_names=selected_class_names,
        image_data=img_base64,
        boxed_image_data=boxed_base64,
        left_score=float(side_scores.get('left', 0.0)),
        right_score=float(side_scores.get('right', 0.0)),
        model_type=model_type
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)