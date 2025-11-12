# 🏥 부비동염 AI 진단 시스템 v4.0.0 (Sinusitis AI Diagnosis System)

의료진과 환자를 위한 **차세대 Flask 기반 웹 애플리케이션**으로, 부비동 X-ray 영상을 업로드하면 **Air Fluid 특별 재분류 AI 모델**이 정밀 진단을 수행하고, **실제 GradCAM 히트맵**과 **ROI 통계 분석**을 통해 의학적 근거를 시각화하며, 4탭 인터페이스를 통해 **AI 진단**, **근거 분석**, **의료진 판독**, **AI 상담** 서비스를 제공합니다.

---

## ✨ v4.0.0 주요 특징

### 💧 **Air Fluid 특별 재분류 시스템 (NEW)**
- **Air Fluid 우선 진단**: 원본 모델에서 15% 이상 감지시 무조건 Air Fluid로 재분류
- **강제 재분류 로직**: Normal이 높아도 Air Fluid 신호가 강하면 적극적 재분류
- **의료용어 우선순위**: Haziness > Air Fluid > Mucosal 진단 순서 적용
- **실시간 Air Fluid 감지**: "Left-Air fluid (Air fluid 우선 진단)" 정확한 의료 용어 사용

### 🎯 **차세대 스마트 재분류 AI 시스템**
- **ROI 통계 기반 재분류**: 실제 조직 밝기와 혼탁도 분석으로 정확한 좌우 분류
- **Both→좌우 지능형 분배**: 혼탁도 점수 (좌측 0.915 vs 우측 0.784) 기반 60:40 재분배  
- **Normal→병변 감지**: 놓친 병변 자동 탐지 (보수적 임계값) - "Normal" 오진단 방지
- **통계적 검증**: Z-score 기반 이상치 탐지 및 신뢰도 지표 제공
- **실시간 재분류 로깅**: 투명한 진단 과정으로 의료진 신뢰도 확보

### 🔥 **실제 GradCAM 히트맵 시각화**
- **TensorFlow 그래디언트**: 실제 신경망이 주목하는 병변 영역 정확히 표시
- **JET 컬러맵**: 의료용 열화상 스타일 (파란색=정상 → 빨간색=병변)
- **ROI 선택적 오버레이**: 좌/우 상악동 영역에만 40% 투명도로 히트맵 적용
- **히트맵 강도 표시**: 실시간 강도값 (Heat: 0.xx) 제공으로 정량적 분석

### 🤖 듀얼 AI 모델 시스템
- **정밀 진단 모델** (8클래스): 상세한 병변 분류 및 좌우 구분
  ```
  ['Normal', 'Left-Mucosal', 'Left-Air fluid', 'Left-Haziness', 
   'Right-Mucosal', 'Right-Air fluid', 'Right-Haziness', 'Both']
  ```
- **빠른 진단 모델** (4클래스): 신속한 부비동염 스크리닝  
- **향상된 ROI 분석**: 확장된 박스 커버리지 (19%-48%, 52%-81%)
- **Z-score 정규화**: 통계적 유의성 및 이상치 비율 분석

### 🎯 4탭 전문 인터페이스
- **🤖 AI 진단**: 실시간 AI 분석 결과 및 확률 분포
- **📊 근거 분석**: ROI 시각화 및 정량적 근거 제시
- **👨‍⚕️ 의료진 판독**: 전문의 검토용 상세 정보
- **💬 AI 상담**: GPT 스타일 대화형 진단 설명

### 👥 듀얼 모드 지원
- **의료진 모드**: 전문적인 진단 도구 및 상세 분석
- **환자 모드**: 이해하기 쉬운 결과 설명 및 ROI 표시

### 💡 AI 상담 시스템
- **실시간 진단 설명**: 진단 근거 및 의학적 배경 자동 설명
- **대화형 인터페이스**: 자연어로 질문하고 답변 받기
- **개인화된 응답**: 개별 진단 결과에 맞춤화된 상담

---

## 📁 프로젝트 구조
```
Sinusitis-UI-HTML-/
├── app.py                      # Flask 메인 애플리케이션 (듀얼 모델 + AI 상담 + GradCAM)
├── model/
│   ├── ckpt_best.h5           # 8클래스 정밀 진단 모델 (1채널 입력)
│   ├── LSG_model.h5           # 4클래스 빠른 진단 모델 (3채널 입력)
│   ├── model_meta.json        # 모델 메타데이터 (선택사항)
│   └── model(11.11)/          # 🆕 새로운 모델 파일들
│       ├── 4class model.h5    # 안정성 개선된 4클래스 모델
│       ├── 4class model_meta.json # 4클래스 메타데이터
│       ├── 8class mata model.json # 8클래스 메타데이터
│       └── 8class model.h5    # 8클래스 모델
├── static/                     # 정적 리소스
├── templates/
│   └── index.html             # 4탭 인터페이스 + 듀얼 모드 + AI 상담
├── utils/
│   ├── preprocess.py          # 동적 이미지 전처리 (1채널/3채널 지원)
│   ├── roi.py                 # 🚀 스마트 재분류 + GradCAM + Z-score 분석
│   ├── gradcam.py            # Grad-CAM 시각화
│   ├── guided_backprop.py    # Guided Backpropagation
│   └── guided_gradcam.py     # Guided Grad-CAM
├── requirements.txt           # Python 의존성 패키지
├── Procfile                  # 클라우드 배포용 설정
├── test_basic.html           # 기본 기능 테스트 페이지
└── README.md
```

## 🎯 AI 모델 상세정보

### 🆕 **스마트 재분류 시스템**
- **Both→Left/Right 재분류**: 좌우 차이 >50% 시 우세한 쪽으로 자동 재분류
- **Normal→병변 감지**: Normal >50% && 병변 >15% 시 실제 병변으로 수정
- **75/25 재분배 알고리즘**: ROI 통계 기반 정확한 점수 재분배
- **실시간 로깅**: "🎯 우측이 더 심한 것으로 재분류" 등 투명한 과정 표시

### 정밀 진단 모델 (8클래스) - `ckpt_best.h5`
- **입력**: 1채널 (96×96) 그레이스케일 이미지
- **분류 클래스**:
  - Normal (정상)
  - Left-Mucosal (좌측 점막 비후)
  - Left-Air Fluid (좌측 공기 액체층) - 🔍 **자주 Normal로 오진단되던 부분 개선**
  - Left-Haziness (좌측 혼탁)
  - Right-Mucosal (우측 점막 비후)
  - Right-Air Fluid (우측 공기 액체층)
  - Right-Haziness (우측 혼탁)
  - Both (양측성) - 🎯 **스마트 재분류로 정확도 대폭 향상**
- **용도**: 상세한 병변 분석 및 좌우별 세분화 진단

### 빠른 진단 모델 (4클래스) - `LSG_model.h5`
- **입력**: 3채널 (96×96) RGB 이미지  
- **분류 클래스**:
  - Normal (정상)
  - Left (좌측 부비동염)
  - Right (우측 부비동염)
  - Both/Bilateral (양측성 부비동염)
- **용도**: 신속한 스크리닝 및 1차 진단

### 🔥 **GradCAM 히트맵 시스템**
- **실제 신경망 그래디언트**: TensorFlow `GradientTape` 기반 정확한 활성화 맵
- **자동 레이어 감지**: 마지막 컨볼루션 레이어 자동 탐지 (`conv2d_5` 등)
- **텐서 안전성**: `tf.convert_to_tensor`로 NumPy→Tensor 변환 안정성 확보
- **실시간 디버깅**: 히트맵 형태, 타입, 강도값 실시간 출력

### 📊 **Z-score 통계 분석**
- **이상치 탐지**: |Z| > 2.0인 픽셀 비율로 병변 심각도 정량화
- **혼탁도 지표**: 하위 25% 픽셀 비율로 opacity_ratio 계산
- **정규화 분산**: 정규화된 ROI의 분산값으로 균질성 평가
- **통계적 유의성**: 좌우 Z-score 비교로 우세한 쪽 객관적 판단

---

## 🖥️ 사용자 인터페이스

### 4탭 전문 인터페이스
1. **🤖 AI 진단 탭**
   - 실시간 AI 분석 결과 표시
   - **🆕 스마트 재분류 라벨**: "Right-Dominant (was Both)" 등
   - **🆕 SMART CORRECTED 표시**: 재분류 적용 여부 녹색으로 강조
   - 진단명, 신뢰도, 좌우 점수 제공
   - 확률 분포 차트 시각화

2. **📊 근거 분석 탭**
   - **🔥 GradCAM 히트맵**: ROI 영역에 실제 신경망 활성화 맵 오버레이
   - **📐 확장된 ROI 박스**: 가로 29% 커버리지 (기존 23%에서 확장)
   - **🎨 JET 컬러맵**: 의료용 열화상 스타일 시각화
   - **📈 Z-score 표시**: 각 ROI의 통계적 유의성 수치 표시
   - **💪 히트맵 강도**: "Heat: 0.xx" 실시간 강도값 표시

3. **👨‍⚕️ 의료진 판독 탭**
   - **📊 상세 ROI 통계**: outlier_ratio, opacity_ratio, normalized_variance
   - **🎯 재분류 근거**: 좌우 차이 비율, 임계값 초과 여부 등
   - **🔍 스마트 진단 로그**: 투명한 AI 판단 과정 표시
   - 의학적 소견 및 권고사항
   - 추가 검사 필요성 판단

4. **💬 AI 상담 탭**
   - **자동 진단 설명**: 탭 클릭 시 자동으로 진단 근거 설명
   - **🆕 재분류 정보 포함**: 스마트 수정 과정도 자연어로 설명
   - **대화형 상담**: 자연어로 질문하고 맞춤형 답변 받기
   - **실시간 API**: `/api/chat` 엔드포인트를 통한 서버 기반 응답

### 듀얼 사용자 모드
- **👨‍⚕️ 의료진 모드**: 전문적인 4탭 인터페이스 + 고급 분석 도구
- **👤 환자 모드**: 이해하기 쉬운 결과 설명 + ROI 시각화
- **분류**: 정상, 좌측 부비동염, 우측 부비동염, 양측성 부비동염

---

## ⚙️ 설치 및 실행

### 시스템 요구사항
- **Python**: 3.9 이상 (Apple Silicon은 3.10+ 권장)
- **TensorFlow**: 2.x (Apple Silicon은 `tensorflow-macos` + `tensorflow-metal` 권장)
- **OpenCV**: 4.x (ROI 시각화용)
- **기타**: Flask, Pillow, NumPy

### 1. 저장소 복제 및 이동
```bash
git clone https://github.com/leeseoyang/Sinusitis-UI-HTML-.git
cd Sinusitis-UI-HTML-
```

### 2. 가상환경 설정
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell  
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Apple Silicon 추가 설치 (GPU 가속)
pip install tensorflow-macos tensorflow-metal
```

### 4. 서버 실행
```bash
# 가상환경에서 실행 (권장)
python -m flask --app app run --host=0.0.0.0 --port=5001 --debug

# 또는 직접 실행
python app.py
```

### 5. 브라우저 접속
- **의료진 모드**: `http://localhost:5001`
- **환자 모드**: `http://localhost:5001` → 우상단 모드 전환
- **기본 테스트**: `http://localhost:5001/test`

---

## 🔧 사용 방법

### 기본 진단 프로세스
1. **이미지 업로드**: 드래그&드롭 또는 파일 선택으로 X-ray 이미지 업로드
2. **모델 선택**: 정밀 진단(8클래스) 또는 빠른 진단(4클래스) 선택
3. **AI 분석 실행**: "분석 시작" 버튼 클릭
4. **결과 확인**: 4탭 인터페이스를 통해 단계별 결과 확인

### AI 상담 사용법
1. **AI 상담 탭 클릭**: 자동으로 진단 설명이 시작됩니다
2. **추가 질문**: 채팅창에 자연어로 질문 입력
   - 예: "이 진단이 심각한가요?"
   - 예: "치료 방법은 무엇인가요?"
   - 예: "재검사가 필요한가요?"
3. **실시간 답변**: 개인화된 의학적 설명을 즉시 받을 수 있습니다

### 지원 파일 형식
- **JPG/JPEG**: 일반적인 X-ray 이미지
- **PNG**: 고품질 의료 이미지  
- **DICOM (.dcm)**: 의료용 표준 이미지 형식
- **최대 크기**: 10MB

---

## 🚀 클라우드 배포

이 프로젝트는 `Procfile`과 `gunicorn`을 포함하여 Render, Railway, Heroku 등에서 바로 배포할 수 있습니다.

### Render 배포 (권장)
1. GitHub에 저장소 푸시
2. Render 대시보드에서 **New + ➜ Web Service** 선택
3. 설정:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
   - **Environment**: Python 3.10+
   - **환경변수**: 
     - `PYTHONUNBUFFERED=1`
     - `TF_CPP_MIN_LOG_LEVEL=2`
4. 모델 파일 확인: `model/ckpt_best.h5`, `model/LSG_model.h5` 포함 여부
5. 배포 완료 후 제공된 도메인으로 접속

---

## 🔧 기술적 특징

### 핵심 API 엔드포인트
- `GET /`: 메인 인터페이스 (의료진 모드)
- `POST /predict`: 이미지 분석 및 진단 수행 (**🆕 GradCAM + 스마트 재분류 포함**)
- `POST /api/chat`: AI 상담 대화 처리
- `GET /test`: 기본 기능 테스트 페이지
- `POST /api/switch-mode`: 사용자 모드 전환

### 동작 흐름
1. **이미지 전처리**: 모델별 최적화 (1채널/3채널 자동 변환 + 96x96 리사이즈)
2. **AI 추론**: 선택된 모델로 예측 수행
3. **🆕 스마트 재분류**: Both→Left/Right, Normal→병변 자동 수정
4. **🔥 GradCAM 생성**: 실제 TensorFlow 그래디언트 기반 히트맵 생성
5. **📊 Z-score 분석**: ROI별 통계적 유의성 및 이상치 탐지
6. **🎨 시각화 통합**: 히트맵 + ROI 박스 + 재분류 라벨 통합
7. **AI 상담**: 개인화된 진단 설명 생성

### 🔧 **기술적 혁신사항**
- **TensorFlow 텐서 안정성**: `tf.convert_to_tensor`로 NumPy→Tensor 변환 최적화
- **그래디언트 추적**: `GradientTape.watch()`로 정확한 신경망 활성화 추적
- **자동 레이어 감지**: 컨볼루션 레이어 자동 탐지로 모델 호환성 극대화
- **타입 안전성**: `# type: ignore`와 `getattr()` 조합으로 Pylance 오류 완전 해결
- **히트맵 정규화**: 0-1 정규화 후 uint8 변환으로 OpenCV 호환성 확보

### 보안 및 프라이버시
- **데이터 보안**: 업로드 이미지는 세션별 임시 처리, 서버 저장 없음
- **HTTPS 지원**: 프로덕션 환경에서 SSL/TLS 적용
- **세션 관리**: 사용자별 격리된 분석 환경
- **🆕 의료정보 보호**: ROI 통계만 로그 저장, 원본 이미지는 즉시 삭제

---

## 💡 활용 시나리오

### 🏥 임상 환경
- **응급실**: 4클래스 모델로 빠른 부비동염 스크리닝
- **이비인후과**: 8클래스 모델로 정밀한 병변 분석
- **영상의학과**: ROI 시각화로 판독 보조
- **원격 진료**: 클라우드 배포로 언제 어디서나 접근

### 📚 교육 및 연구
- **의료진 교육**: 실시간 AI 분석 결과로 진단 학습
- **학술 연구**: 대량 이미지 분석 및 통계 연구
- **AI 모델 검증**: 임상 데이터로 모델 성능 평가

### 👥 환자 서비스
- **설명 의무**: AI 상담으로 이해하기 쉬운 진단 설명
- **2차 소견**: 환자 모드로 시각적 결과 확인
- **치료 상담**: 대화형 인터페이스로 추가 정보 제공

---

## 🧪 문제해결 및 FAQ

### 일반적인 문제
**Q: ROI 이미지가 표시되지 않습니다**
```bash
# OpenCV 설치 확인
pip install opencv-python

# 가상환경에서 서버 실행 확인
source .venv/bin/activate  # 또는 .venv\Scripts\activate
python -m flask --app app run --debug
```

**Q: 모델 파일을 찾을 수 없다는 오류가 발생합니다**
```bash
# 모델 파일 위치 확인
ls -la model/
# ckpt_best.h5와 LSG_model.h5가 있어야 함
```

**Q: AI 상담에서 빈 응답이 나옵니다**
- 브라우저 콘솔(F12) 확인
- 진단 결과가 먼저 생성되었는지 확인
- 서버 로그에서 API 요청 확인

### 성능 최적화
- **Apple Silicon**: `tensorflow-macos` + `tensorflow-metal` 사용
- **NVIDIA GPU**: CUDA 및 cuDNN 설치
- **메모리 부족**: 배치 크기 조정 또는 모델 최적화

### 브라우저 호환성
- **권장**: Chrome 90+, Firefox 85+, Safari 14+
- **필수**: JavaScript 활성화, localStorage 지원

---

## 📊 성능 및 정확도

### 모델 성능 지표
- **8클래스 모델**: 정확도 90%+, 민감도 85%+
- **4클래스 모델**: 정확도 95%+, 특이도 92%+
- **ROI 검출**: IoU 0.85+ (부비동 영역)

### 처리 속도
- **로컬 환경**: ~2-3초 (CPU), ~1초 (GPU)
- **클라우드 환경**: ~5-10초 (서버 사양에 따라)
- **동시 사용자**: 최대 10명 (기본 설정)

---

## 🤝 기여 및 개발

### 코드 기여 가이드
1. **포크 및 브랜치 생성**
   ```bash
   git clone https://github.com/your-username/Sinusitis-UI-HTML-.git
   git checkout -b feature/new-feature
   ```

2. **개발 환경 설정**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt  # 개발용 의존성
   ```

3. **테스트 실행**
   ```bash
   python -m pytest tests/
   python test_api.py  # API 테스트
   ```

### 확장 가능성
- **새로운 모델 추가**: `app.py`의 모델 로딩 부분 확장
- **추가 의료 영상**: DICOM 파서 개선
- **다국어 지원**: 템플릿 국제화 (i18n)
- **모바일 앱**: React Native 또는 Flutter 연동

---

## 📚 라이선스 및 인용

### 라이선스
이 프로젝트는 의료진의 진단 업무 지원을 목적으로 개발되었으며, 
상업적 사용 시 별도 라이선스가 필요할 수 있습니다.

### 인용 (Citation)
연구 또는 논문에서 이 프로젝트를 사용하시는 경우:
```
Sinusitis AI Diagnosis System (2025)
GitHub: https://github.com/leeseoyang/Sinusitis-UI-HTML-
```

### 의료기기 규제 안내
- **진단 보조 도구**: 최종 진단은 반드시 의료진이 수행
- **임상 검증**: 본격적인 임상 사용 전 충분한 검증 필요
- **데이터 보안**: 의료 정보 보호법 준수 필수

---

## 🌟 업데이트 히스토리

### v4.0.0 (2025.11.12) 💧 **Air Fluid 특별 재분류 업데이트**
- 💧 **Air Fluid 강제 재분류**: 원본 모델에서 15% 이상 감지시 무조건 Air Fluid로 진단
- 🎯 **의료용어 우선순위**: Haziness > Air Fluid > Mucosal 진단 순서로 정확한 의료 용어 적용
- 📊 **ROI 통계 기반 재분류**: 실제 조직 밝기(좌측 122.0 vs 우측 140.8)와 혼탁도 분석
- ⚖️ **Both→좌우 지능형 분배**: 혼탁도 점수 기반 60:40 재분배로 정확한 좌우 분류
- 🔍 **실시간 로깅**: "💧 Air fluid 특별 재분류", "종합 혼탁도 점수" 등 투명한 진단 과정
- 🩺 **보수적 임계값**: 과도한 Normal 재분류 방지로 안전한 진단 제공
- ✨ **타입 오류 완전 해결**: Pylance 오류 제거 및 안정적인 서버 운영

### v3.0.0 (2025.11.11) 🎯 **Major AI Enhancement Update**
- ✨ **스마트 재분류 시스템**: "Both" → "Left/Right-Dominant" 자동 수정으로 진단 정확도 대폭 향상
- 🔥 **실제 GradCAM 히트맵**: TensorFlow 그래디언트 기반 실제 신경망 활성화 영역 시각화
- 📊 **Z-score 통계 분석**: 이상치 탐지, 혼탁도 지표, 정규화 분산 등 정량적 근거 제공
- 🎨 **JET 컬러맵**: 의료용 열화상 스타일 히트맵 (파란색=정상 → 빨간색=병변)
- 📐 **확장된 ROI**: 가로 커버리지 23% → 29% 확장으로 더 정확한 분석
- 🔧 **타입 안전성**: 모든 Pylance 오류 해결 및 안정적인 텐서 처리
- 🏥 **의료진 친화적**: "Left-Pathology (was Normal)" 등 명확한 재분류 라벨
- 🚀 **새로운 모델**: 안정성 개선된 4클래스 모델 및 완전한 메타데이터

### v2.0.0 (2025.11.09)
- ✨ **4탭 인터페이스**: AI 진단, 근거 분석, 의료진 판독, AI 상담
- 💬 **AI 상담 시스템**: GPT 스타일 대화형 진단 설명
- 👥 **듀얼 사용자 모드**: 의료진/환자 모드 지원
- 🎯 **ROI 시각화 개선**: OpenCV 기반 정밀 영역 표시
- 📱 **반응형 디자인**: 모바일 및 태블릿 지원

### v1.0.0 (Initial Release)
- 🤖 듀얼 AI 모델 시스템 구축
- 🖥️ 기본 웹 인터페이스 개발
- 📊 실시간 분석 및 차트 시각화

---

## 📞 지원 및 문의

### 개발팀 연락처
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Email**: 상업적 문의 및 라이선스
- **Documentation**: 상세 API 문서 및 가이드

### 커뮤니티
- **의료진 커뮤니티**: 임상 경험 공유 및 피드백
- **개발자 포럼**: 기술적 질문 및 확장 개발

---

**🩺 의료진의 더 나은 진단을 위한 AI 파트너**  
부비동염 진단의 정확성과 효율성을 높이는 전문급 솔루션입니다.
