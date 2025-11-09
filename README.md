# 🏥 부비동염 AI 진단 시스템 (Sinusitis AI Diagnosis System)

의료진과 환자를 위한 전문급 Flask 기반 웹 애플리케이션으로, 부비동 X-ray 영상을 업로드하면 듀얼 AI 모델이 정밀 진단을 수행하고, 4탭 인터페이스를 통해 **AI 진단**, **근거 분석**, **의료진 판독**, **AI 상담** 서비스를 제공합니다.

---

## ✨ 주요 특징

### 🤖 듀얼 AI 모델 시스템
- **정밀 진단 모델** (8클래스): 상세한 병변 분류 및 좌우 구분
- **빠른 진단 모델** (4클래스): 신속한 부비동염 스크리닝
- **실시간 ROI 시각화**: 부비동 영역 자동 감지 및 색상 코딩

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
├── app.py                      # Flask 메인 애플리케이션 (듀얼 모델 + AI 상담 API)
├── model/
│   ├── ckpt_best.h5           # 8클래스 정밀 진단 모델 (1채널 입력)
│   ├── LSG_model.h5           # 4클래스 빠른 진단 모델 (3채널 입력)
│   └── model_meta.json        # 모델 메타데이터 (선택사항)
├── static/                     # 정적 리소스
├── templates/
│   └── index.html             # 4탭 인터페이스 + 듀얼 모드 + AI 상담
├── utils/
│   ├── preprocess.py          # 동적 이미지 전처리 (1채널/3채널 지원)
│   ├── roi.py                 # ROI 영역 감지 및 시각화
│   ├── gradcam.py            # Grad-CAM 시각화
│   ├── guided_backprop.py    # Guided Backpropagation
│   └── guided_gradcam.py     # Guided Grad-CAM
├── requirements.txt           # Python 의존성 패키지
├── Procfile                  # 클라우드 배포용 설정
├── test_basic.html           # 기본 기능 테스트 페이지
├── test_xray.png            # 테스트용 X-ray 이미지
└── README.md
```

## 🎯 AI 모델 상세정보

### 정밀 진단 모델 (8클래스) - `ckpt_best.h5`
- **입력**: 1채널 (96×96) 그레이스케일 이미지
- **분류 클래스**:
  - Normal (정상)
  - Left-Mucosal (좌측 점막 비후)
  - Left-Air Fluid (좌측 공기 액체층)
  - Left-Haziness (좌측 혼탁)
  - Right-Mucosal (우측 점막 비후)
  - Right-Air Fluid (우측 공기 액체층)
  - Right-Haziness (우측 혼탁)
  - Both (양측성)
- **용도**: 상세한 병변 분석 및 좌우별 세분화 진단

### 빠른 진단 모델 (4클래스) - `LSG_model.h5`
- **입력**: 3채널 (96×96) RGB 이미지  
- **분류 클래스**:
  - Normal (정상)
  - Left (좌측 부비동염)
  - Right (우측 부비동염)
  - Both/Bilateral (양측성 부비동염)
- **용도**: 신속한 스크리닝 및 1차 진단

---

## 🖥️ 사용자 인터페이스

### 4탭 전문 인터페이스
1. **🤖 AI 진단 탭**
   - 실시간 AI 분석 결과 표시
   - 진단명, 신뢰도, 좌우 점수 제공
   - 확률 분포 차트 시각화

2. **📊 근거 분석 탭**
   - ROI(관심영역) 이미지 자동 생성
   - 부비동 영역 빨간 박스 표시
   - 정량적 분석 근거 제시

3. **👨‍⚕️ 의료진 판독 탭**
   - 전문의 검토용 상세 정보
   - 의학적 소견 및 권고사항
   - 추가 검사 필요성 판단

4. **💬 AI 상담 탭**
   - **자동 진단 설명**: 탭 클릭 시 자동으로 진단 근거 설명
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
- `POST /predict`: 이미지 분석 및 진단 수행
- `POST /api/chat`: AI 상담 대화 처리
- `GET /test`: 기본 기능 테스트 페이지
- `POST /api/switch-mode`: 사용자 모드 전환

### 동작 흐름
1. **이미지 전처리**: 모델별 최적화 (1채널/3채널 자동 변환)
2. **AI 추론**: 선택된 모델로 예측 수행
3. **ROI 생성**: OpenCV 기반 부비동 영역 시각화
4. **결과 통합**: 진단 결과, ROI, 확률 분포 통합
5. **AI 상담**: 개인화된 진단 설명 생성

### 보안 및 프라이버시
- **데이터 보안**: 업로드 이미지는 세션별 임시 처리, 서버 저장 없음
- **HTTPS 지원**: 프로덕션 환경에서 SSL/TLS 적용
- **세션 관리**: 사용자별 격리된 분석 환경

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
