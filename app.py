"""
Flask 앱 (부비동염 분류 데모)
 - 외부 의존성(TensorFlow, Matplotlib 등)은 지연 로딩하여 개발 환경에서의 임포트 오류를 완화
 - Grad-CAM 레이어 자동 탐색 및 예외 처리 강화
"""

import os
import io
import base64
import json
from typing import Tuple, List, Any, Optional, Dict
from flask import Flask, render_template, request

# 백엔드/런타임 설정 (전역)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 줄이기
os.environ['MPLBACKEND'] = 'Agg'          # Matplotlib 비-GUI 백엔드

# Flask 앱 초기화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

_default_class_names = ['Normal', 'Right-Mucosal', 'Right-Air fluid', 'Right-Haziness',
                        'Left-Mucosal', 'Left-Air fluid', 'Left-Haziness', 'Both']

def _load_model_and_classes() -> Tuple[Optional[Any], List[str]]:
    """TensorFlow 및 모델을 지연 로딩하고 클래스 이름 메타를 안전하게 읽는다."""
    loaded_model: Optional[Any] = None
    class_names: List[str] = _default_class_names
    try:
        from tensorflow import keras  # type: ignore
        loaded_model = keras.models.load_model('model/model.h5', compile=False)  # type: ignore
        # 메타 파일에서 클래스 이름 읽기 (선택적)
        try:
            with open('model/model_meta.json', 'r', encoding='utf-8') as meta_file:
                meta = json.load(meta_file)
            class_names = meta.get('class_names', _default_class_names)
            # 모델 출력 크기와 길이 불일치 시 기본값으로 롤백
            if hasattr(loaded_model, 'output_shape') and len(class_names) != loaded_model.output_shape[-1]:  # type: ignore
                class_names = _default_class_names
        except (FileNotFoundError, json.JSONDecodeError):
            class_names = _default_class_names
    except Exception:
        # TensorFlow/모델 로딩 실패 시 None 유지, 기본 클래스 사용
        loaded_model = None
    return loaded_model, class_names  # type: ignore

# 앱 시작 시 한 번 시도 (실패하더라도 서버는 뜨고, 예측 시점에 안내)
model, class_names = _load_model_and_classes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  # type: ignore
    # 지연 임포트 (개발 환경에서 불필요한 임포트 오류 완화)
    from PIL import Image
    import numpy as np
    from utils.preprocess import preprocess_and_correct  # type: ignore

    # 🔹 이미지 로딩/검증
    if 'image' not in request.files:
        # 업로드 누락 처리
        return render_template('index.html', prediction=None, class_names=None, probs=None, image_data=None,
                               boxed_image_data=None, left_score=None, right_score=None)
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction=None, class_names=None, probs=None, image_data=None,
                               boxed_image_data=None, left_score=None, right_score=None)

    try:
        image = Image.open(file.stream)
    except Exception:
        # 잘못된 이미지 형식
        return render_template('index.html', prediction=None, class_names=None, probs=None, image_data=None,
                               boxed_image_data=None, left_score=None, right_score=None)

    # 🔹 전처리 (자동으로 그레이스케일 변환)
    image_for_model, corrected_pil = preprocess_and_correct(image)  # type: ignore

    # 모델 준비 확인
    if model is None:
        # 모델 미로딩/TF 미설치 상태에서는 입력/전처리만 표시
        img_byte_arr = io.BytesIO()
        corrected_pil.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return render_template('index.html', prediction=None, class_names=None, probs=None,
                               image_data=img_base64, boxed_image_data=None, left_score=None, right_score=None)

    # 🔹 예측 (이미 (96, 96, 1) 형태로 전처리됨)
    preds = model.predict(image_for_model[np.newaxis, ...], batch_size=1)[0]
    pred_index = int(np.argmax(preds))
    pred_class = class_names[pred_index]
    confidence = float(np.max(preds) * 100.0)

    # 🔹 박스 기반 자동 추적 오버레이 생성
    boxed_base64 = None
    side_scores = {"left": 0.0, "right": 0.0}
    try:
        import cv2
        from utils.roi import summarize_side_scores, draw_boxes_on_image
        import numpy as np

        side_scores: Dict[str, float] = summarize_side_scores(preds, class_names)
        bgr = cv2.cvtColor(np.array(corrected_pil.convert('L')), cv2.COLOR_GRAY2BGR)
        boxed_bgr = draw_boxes_on_image(bgr.copy(), side_scores, label=pred_class, conf=confidence)  # type: ignore

        buf2 = io.BytesIO()
        img_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)  # type: ignore
        Image.fromarray(img_rgb).save(buf2, format='PNG')  # type: ignore
        boxed_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    except Exception:
        boxed_base64 = None

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
        boxed_image_data=boxed_base64,
        left_score=float(side_scores.get('left', 0.0)),
        right_score=float(side_scores.get('right', 0.0))
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_flag = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_flag)
