# 🧠 부비동염 판별 시스템 (Sinusitis Detection UI)

Flask 기반 웹 애플리케이션으로 부비동(부비동염) X-ray 이미지를 업로드하면 학습된 TensorFlow 모델이 8개 클래스 중 하나를 예측하고, 좌/우 상악동 영역 박스를 통해 예측 결과를 시각화합니다.

---

## ✨ 주요 특징
- 이미지 업로드 직후 프리뷰, 결과 배지, 차트 표시
- 예측 확률을 Chart.js 막대 차트로 가시화
- 좌/우 상악동 박스를 색상(녹→노→적)으로 표시해 예측 강도를 직관적으로 전달
- 로고 교체, PDF 내보내기 등 UI 편의 기능 포함

---

## 📁 폴더 구조
```
Sinusitis-UI-HTML-/
├── app.py                      # Flask 엔트리 포인트
├── model/
│   ├── model.h5                # 학습된 Keras 모델 (필수)
│   ├── ckpt_best.h5            # 추가 체크포인트(선택)
│   └── model_meta.json         # 클래스 이름 등 메타데이터(선택)
├── static/
│   └── logo.png                # 기본 로고 자산
├── templates/
│   └── index.html              # 메인 웹 페이지 템플릿
├── utils/
│   ├── gradcam.py             # Grad-CAM 생성 유틸(보존용)
│   ├── guided_backprop.py     # Guided Backpropagation(보존용)
│   ├── guided_gradcam.py      # Guided Grad-CAM 결합 유틸(보존용)
│   ├── preprocess.py          # PIL 기반 전처리 파이프라인
│   └── roi.py                 # 좌/우 상악동 박스 계산 및 그리기
├── UI 데모 코드.txt
└── README.md
```

---

## ⚙️ 요구 사항
- Python 3.9 이상 (Apple Silicon의 경우 3.10+ 권장)
- TensorFlow 2.x (Apple Silicon은 `tensorflow-macos`, `tensorflow-metal` 조합 권장)
- Flask, Pillow, NumPy, OpenCV, Chart.js(프런트엔드)

### 가상환경 생성 및 의존성 설치
macOS 또는 Linux 기준:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell 기준:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

작업이 끝난 뒤에는 다음 명령으로 가상환경을 종료합니다.
```bash
deactivate
```

> `requirements.txt`에는 TensorFlow, Flask, Pillow, NumPy, OpenCV 등 프로젝트 실행에 필요한 패키지가 모두 정의되어 있습니다. Apple Silicon에서는 `pip install tensorflow-macos tensorflow-metal`을 별도로 실행하면 GPU 가속을 활용할 수 있습니다.

---

## 🚀 실행 방법
1. 학습된 모델 파일을 `model/model.h5` 경로에 배치합니다.
2. (선택) 커스텀 클래스 이름을 사용하려면 `model/model_meta.json`에 `{"class_names": [...]}` 형태로 저장합니다.
3. 가상환경이 활성화된 상태에서 서버 실행:
    ```bash
    python app.py
    ```
4. 브라우저에서 `http://127.0.0.1:5001` 접속  
    (기본적으로 `0.0.0.0:5001`에서 실행되며, 필요 시 포트 또는 디버그 설정을 변경하세요.)

---

## ☁️ 클라우드 배포 (Render 예시)
터미널이 열려 있지 않아도 지속적으로 서비스를 제공하려면 클라우드 호스팅과 프로덕션 WSGI 서버(`gunicorn`)를 이용하는 것이 안전합니다. 이 저장소에는 `Procfile`과 `gunicorn` 의존성이 포함되어 있으므로 Render, Railway, Heroku 등에서 바로 사용할 수 있습니다. Render 기준 단계는 다음과 같습니다.

1. 이 저장소를 GitHub에 푸시합니다.
2. Render 대시보드에서 **New + ➜ Web Service**를 선택하고 GitHub 저장소를 연결합니다.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Environment: Python 3.10 이상을 선택하고, 필요 시 `PYTHONUNBUFFERED=1`, `TF_CPP_MIN_LOG_LEVEL=2` 같은 환경 변수를 추가합니다.
6. `model/model.h5`와 `model/model_meta.json` 파일이 저장소에 포함되어 있는지 확인합니다. 보안상 모델 파일을 별도 스토리지에 보관하고 싶다면 Render의 **Persistent Disk** 또는 S3 버킷을 사용한 뒤, 런치 스크립트에서 다운로드하도록 구성할 수 있습니다.
7. 배포가 완료되면 Render가 발급한 도메인으로 접속합니다. 기본 포트는 Render가 제공한 값이 되며, Flask가 `PORT` 환경 변수를 읽어 동적으로 바인딩할 수 있도록 `app.py`에서 `port = int(os.environ.get("PORT", 5001))` 형태로 처리되어 있습니다.

> Railway, Heroku 등 다른 PaaS에서도 동일하게 `gunicorn app:app` 스타트 커맨드와 `Procfile`을 사용해 배포할 수 있습니다. Docker 환경을 쓰고 싶다면 `FROM python:3.11-slim` 이미지로 시작해 `pip install -r requirements.txt` 후 `gunicorn app:app --bind 0.0.0.0:$PORT`를 CMD로 지정하면 됩니다.

---

## 🔍 동작 개요
### `app.py`
- 업로드된 이미지를 Pillow로 로드하고 `utils.preprocess.preprocess_and_correct`로 (96, 96, 1) 형태로 정규화합니다.
- TensorFlow 모델 예측 결과를 기반으로 최대 확률 클래스와 확률값을 계산합니다.
- `utils.roi.summarize_side_scores`가 좌/우/양측 스코어를 추출하고, `utils.roi.draw_boxes_on_image`가 색상 박스를 그립니다.
- 박스가 그려진 이미지는 Base64로 인코딩되어 템플릿으로 전달됩니다.
- `model_meta.json`에 클래스 배열이 있다면 이를 사용하고, 없으면 기본 8개 클래스 이름을 사용합니다.

### `templates/index.html`
- 업로드 드롭존, 결과 배지, 차트, PDF 내보내기 등 UI 구성.
- Chart.js를 사용해 클래스별 확률을 가로 막대형으로 렌더링.
- 서버에서 전달한 Base64 이미지를 사용해 입력 이미지와 좌/우 박스 시각화를 즉시 표시.
- 로고 크기 조절, 로고 교체 기능, 접근성 메시지(`aria-live`) 등을 포함.

### `utils/`
- `preprocess.py`: PIL 이미지를 단일 채널, 0~1 범위로 정규화.
- `roi.py`: 좌/우 상악동 박스 좌표 계산, 색상 매핑, 텍스트 주석 처리.
- `gradcam.py`, `guided_backprop.py`, `guided_gradcam.py`: 기존 Grad-CAM 파이프라인(현재 UI에서는 비활성화 상태) 보존용.

---

## 🧪 테스트 팁
- 정상 실행 후 예시 X-ray 이미지를 업로드해 차트와 Guided Grad-CAM이 제대로 표시되는지 확인합니다.
- 차트가 렌더링되지 않는다면 브라우저 콘솔에서 `class_names` 또는 `probs` 직렬화 오류를 확인하세요.
- 서버 콘솔에 TensorFlow 관련 경고가 뜰 수 있으나, 필요 시 `TF_CPP_MIN_LOG_LEVEL=2`로 줄일 수 있습니다. (`app.py`에서 이미 설정)

---

## 📚 참고 사항
- 현재 모델은 8개 클래스(`Normal`, `Right-Mucosal`, `Right-Air fluid`, `Right-Haziness`, `Left-Mucosal`, `Left-Air fluid`, `Left-Haziness`, `Both`) 예측을 전제로 합니다.
- Guided Grad-CAM의 `last_conv_layer_name`은 기본적으로 `conv2d_5`로 설정되어 있으나, 모델 구조에 따라 조정이 필요할 수 있습니다.
- PDF 내보내기 기능은 클라이언트 측 렌더링을 기반으로 하므로, 차트가 표시된 이후 실행해야 합니다.

---

## 🤝 기여
이 저장소는 빠른 프로토타입용으로 작성되었습니다. 이슈나 개선 아이디어가 있다면 자유롭게 제안해 주세요!

---
