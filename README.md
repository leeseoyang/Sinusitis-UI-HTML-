# 🏥 부비동염 AI 진단 시스템 (Sinusitis AI Diagnosis System)

의료진을 위한 Flask 기반 웹 애플리케이션으로, 부비동 X-ray 영상을 업로드하면 학습된 AI 모델이 부비동염 상태를 정확히 진단하고 ROI(관심영역) 시각화를 통해 결과를 직관적으로 제공합니다.

---

## ✨ 주요 특징
- **듀얼 AI 모델**: 정밀 진단(8클래스) + 빠른 진단(4클래스) 지원
- **의료진 친화적 UI**: 직관적이고 전문적인 의료용 인터페이스
- **ROI 시각화**: 부비동 영역 자동 감지 및 색상 코딩 표시
- **실시간 분석**: 이미지 업로드 즉시 AI 진단 결과 제공
- **동적 전처리**: 모델별 최적화된 이미지 처리 (1채널/3채널 자동 변환)
- **결과 내보내기**: PDF 형태로 진단 결과 저장 가능

---

## 📁 폴더 구조
```
Sinusitis-UI-HTML-/
├── app.py                      # Flask 메인 애플리케이션
├── model/
│   ├── ckpt_best.h5           # 8클래스 정밀 진단 모델 (1채널 입력)
│   ├── LSG_model.h5           # 4클래스 빠른 진단 모델 (3채널 입력)
│   └── model_meta.json        # 모델 메타데이터 (선택사항)
├── static/
│   └── logo.png               # 기본 로고 파일
├── templates/
│   └── index.html             # 의료진 친화적 웹 인터페이스
├── utils/
│   ├── preprocess.py          # 동적 이미지 전처리 (1채널/3채널 지원)
│   ├── roi.py                 # ROI 영역 감지 및 시각화
│   ├── gradcam.py            # Grad-CAM 시각화 (보존용)
│   ├── guided_backprop.py    # Guided Backpropagation (보존용)
│   └── guided_gradcam.py     # Guided Grad-CAM (보존용)
├── requirements.txt           # Python 의존성 패키지
├── Procfile                  # 클라우드 배포용 설정
└── README.md
```

## 🎯 지원 모델

### 정밀 진단 모델 (8클래스)
- **파일**: `ckpt_best.h5`
- **입력**: 1채널 (96×96) 그레이스케일
- **분류**: 정상, 좌측/우측 점막 비후, 좌측/우측 액체 저류, 좌측/우측 혼탁, 양측성

### 빠른 진단 모델 (4클래스)  
- **파일**: `LSG_model.h5`
- **입력**: 3채널 (96×96) RGB
- **분류**: 정상, 좌측 부비동염, 우측 부비동염, 양측성 부비동염

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
1. **모델 파일 준비**: 
   - `model/ckpt_best.h5` (8클래스 모델)
   - `model/LSG_model.h5` (4클래스 모델)
2. **가상환경 활성화 및 서버 실행**:
    ```bash
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    python app.py
    ```
3. **브라우저 접속**: `http://127.0.0.1:5001`
4. **사용법**:
   - 파일 선택 버튼을 클릭하여 X-ray 이미지 업로드
   - 진단 모델 선택 (정밀 진단 또는 빠른 진단)
   - AI 진단 시작 버튼 클릭
   - 실시간 진단 결과 및 ROI 시각화 확인

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

### 핵심 기술
- **듀얼 모델 시스템**: 8클래스와 4클래스 모델을 동시 지원하여 상황에 맞는 진단 제공
- **동적 전처리**: 모델별 입력 요구사항에 따라 자동으로 1채널 또는 3채널로 이미지 변환
- **ROI 자동 감지**: 부비동 영역을 자동으로 탐지하고 시각적으로 표시
- **의료진 최적화**: 직관적인 의료용 UI/UX로 임상 환경에서 바로 사용 가능

### `app.py` 주요 기능
- Flask 웹 서버 및 라우팅 처리
- 듀얼 모델 로딩 및 동적 모델 선택
- 이미지 전처리 및 AI 추론 실행
- ROI 시각화 및 결과 통합 처리
- Base64 인코딩을 통한 실시간 이미지 전송

### `templates/index.html` UI 특징
- 의료진 친화적인 전문 디자인 (의료용 색상 체계)
- 간단한 파일 선택 버튼 (복잡한 드래그&드롭 제거)
- 실시간 차트 시각화 (Chart.js)
- PDF 진단서 내보내기 기능
- 접근성 최적화 (`aria-live`, 스크린 리더 지원)

### `utils/` 유틸리티
- `preprocess.py`: 모델별 최적화된 이미지 전처리 (1채널/3채널 동적 변환)
- `roi.py`: 부비동 영역 감지, 색상 코딩, 점수 계산
- 기타: Grad-CAM 관련 모듈들 (향후 고도화용 보존)

---

## 🧪 테스트 및 문제해결
- **정상 실행 확인**: X-ray 이미지 업로드 후 차트와 ROI 시각화가 정상 표시되는지 확인
- **모델 호환성**: 8클래스 모델은 그레이스케일, 4클래스 모델은 RGB 입력 사용
- **브라우저 호환성**: Chrome, Firefox, Safari 등 모던 브라우저에서 테스트 완료
- **성능 최적화**: TensorFlow Metal(Apple Silicon) 또는 CUDA(NVIDIA) 가속 권장

## 💡 사용 시나리오
1. **응급실**: 빠른 진단 모델로 신속한 부비동염 스크리닝
2. **이비인후과**: 정밀 진단 모델로 상세한 병변 분석
3. **영상의학과**: ROI 시각화를 통한 판독 보조
4. **의료진 교육**: 실시간 AI 분석 결과를 활용한 교육 도구

---

## 📚 참고 사항
- **모델 정확도**: 임상 검증된 데이터셋으로 학습된 고성능 AI 모델 사용
- **데이터 보안**: 업로드된 이미지는 서버에 저장되지 않으며 세션 종료 시 자동 삭제
- **호환성**: DICOM, JPG, PNG 등 다양한 의료 영상 포맷 지원
- **확장성**: 추가 모델 통합 및 기능 확장 가능한 모듈형 구조

---

## 🤝 기여 및 문의
이 프로젝트는 의료진의 진단 업무를 지원하기 위해 개발되었습니다. 
개선 사항이나 문의사항이 있으시면 언제든 연락해 주세요!

---
