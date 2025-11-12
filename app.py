# type: ignore
from __future__ import annotations

"""
Flask 웹앱: 부비동염 X-ray 이미지 분류
8클래스 모델 (ckpt_best.h5) + 4클래스 모델 (LSG_model.h5) 지원
"""
import json
import io
import base64
from typing import Dict, Optional, Any, TYPE_CHECKING
from flask import Flask, render_template, request, jsonify

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
        import tensorflow as tf  # type: ignore
        
        # TensorFlow 호환성 설정
        tf.compat.v1.disable_eager_execution = lambda: None  # type: ignore
        
        # 8클래스 모델 로딩 (기존 파일 사용 - 호환성 문제로)
        try:
            model_8class = keras.models.load_model('model/ckpt_best.h5', compile=False)  # type: ignore
            print("✅ 8클래스 모델 (model/ckpt_best.h5) 로딩 성공")
        except Exception as e:
            print(f"❌ 8클래스 모델 로딩 실패: {e}")
            model_8class = None
        
        # 4클래스 모델 로딩 (여러 파일 시도)
        try:
            # 첫 번째 시도: 4class model.h5 (더 안정적)
            model_4class = keras.models.load_model('model/model(11.11)/4class model.h5', compile=False)  # type: ignore
            print("✅ 4클래스 모델 (model(11.11)/4class model.h5) 로딩 성공")
        except Exception as e:
            print(f"❌ 4class model.h5 로딩 실패: {e}")
            try:
                # 안전한 로딩 방법 시도 (custom_objects 사용)
                custom_objects = {'Conv2D': tf.keras.layers.Conv2D}  # type: ignore
                model_4class = keras.models.load_model(  # type: ignore
                    'model/model(11.11)/4class mata model.h5', 
                    compile=False,
                    custom_objects=custom_objects
                )
                print("✅ 4클래스 모델 (4class mata model.h5) 커스텀 로딩 성공")
            except Exception as e2:
                print(f"❌ 4클래스 모델 로딩 완전 실패 - 호환성 문제: {e2}")
                print("⚠️ 8클래스 모델만 사용하여 계속 진행합니다.")
                model_4class = None
        
        # 메타 파일에서 클래스 이름 읽기 (선택적)
        try:
            # 8클래스 모델 메타데이터 로딩
            with open('model/model(11.11)/8class mata model.json', 'r', encoding='utf-8') as meta_file_8:
                meta_8 = json.load(meta_file_8)
            class_names_8 = meta_8.get('class_names', _default_class_names_8)
            print(f"✅ 8클래스 메타데이터 로딩: {class_names_8}")
            
            # 4클래스 모델 메타데이터 로딩
            with open('model/model(11.11)/4class model_meta.json', 'r', encoding='utf-8') as meta_file_4:
                meta_4 = json.load(meta_file_4)
            class_names_4 = meta_4.get('class_names', _default_class_names_4)
            print(f"✅ 4클래스 메타데이터 로딩: {class_names_4}")
            
            # 클래스 수 검증
            if len(class_names_8) != 8:
                print(f"⚠️ 8클래스 모델 클래스 수 불일치: {len(class_names_8)}개, 기본값 사용")
                class_names_8 = _default_class_names_8
                
            if len(class_names_4) != 4:
                print(f"⚠️ 4클래스 모델 클래스 수 불일치: {len(class_names_4)}개, 기본값 사용")
                class_names_4 = _default_class_names_4
            
            print(f"✅ 메타데이터 로딩 성공 - 8클래스: {len(class_names_8)}개, 4클래스: {len(class_names_4)}개")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ 메타데이터 로딩 실패, 기본값 사용: {e}")
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

@app.route('/test')
def test_basic():
    """기본 기능 테스트 페이지"""
    with open('test_basic.html', 'r', encoding='utf-8') as f:
        return f.read()

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
    # 모델의 입력 형태를 확인하여 채널 수 동적 결정
    if model_type == '4class' and model_4class is not None:
        try:
            # 모델의 입력 형태 확인
            input_shape = model_4class.input_shape
            print(f"🔍 4클래스 모델 입력 형태: {input_shape}")
            if len(input_shape) >= 4 and input_shape[-1] == 3:
                channels = 3  # 3채널 입력
                print("📊 4클래스 모델: 3채널 RGB 입력 감지")
            else:
                channels = 1  # 1채널 입력
                print("📊 4클래스 모델: 1채널 그레이스케일 입력 감지")
        except Exception as e:
            print(f"⚠️ 모델 입력 형태 확인 실패: {e}, 기본값 3채널 사용")
            channels = 3  # 기본값으로 3채널 사용
    else:
        channels = 1  # 8클래스 모델은 1채널 사용
    
    image_for_model, corrected_pil = preprocess_and_correct(image, channels=channels)  # type: ignore

    # 모델 선택 및 준비 확인
    if model_type == '4class':
        selected_model = model_4class
        selected_class_names = class_names_4
    else:
        selected_model = model_8class
        selected_class_names = class_names_8
    
    if selected_model is None:
        # 4클래스 모델이 없을 때 8클래스 모델로 대체
        if model_type == '4class' and model_8class is not None:
            print("⚠️ 4클래스 모델 미사용 - 8클래스 모델로 대체")
            selected_model = model_8class
            selected_class_names = class_names_8
            model_type = '8class'  # UI에서 표시용
        else:
            # 모든 모델이 없는 경우
            img_byte_arr = io.BytesIO()
            corrected_pil.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            return render_template('index.html', prediction=None,
                                   class_names_8=class_names_8, class_names_4=class_names_4,
                                   probs=None, image_data=img_base64, boxed_image_data=None, 
                                   left_score=None, right_score=None, model_type=model_type,
                                   error_msg=f'{model_type} 모델을 찾을 수 없습니다. TensorFlow 호환성 문제가 있을 수 있습니다.')

    # 예측 (이미 (96, 96, 1) 형태로 전처리됨)
    preds = selected_model.predict(image_for_model[np.newaxis, ...], batch_size=1)[0]  # type: ignore
    
    # 4분류 모델이 8개 출력을 가지는 경우 매핑 처리
    if model_type == '4class' and len(preds) == 8:
        print(f"🔄 4분류 모델 8→4 매핑 적용: {preds.shape}")
        # 8클래스를 4클래스로 매핑: [Normal, Left, Right, Both]
        mapped_preds = np.zeros(4)
        mapped_preds[0] = preds[0]  # Normal
        mapped_preds[1] = preds[1] + preds[2] + preds[3]  # Left (Mucosal + Air fluid + Haziness)
        mapped_preds[2] = preds[4] + preds[5] + preds[6]  # Right (Mucosal + Air fluid + Haziness)
        mapped_preds[3] = preds[7]  # Both
        preds = mapped_preds
        print(f"✅ 4분류 매핑 완료: {preds}")
    
    pred_index = int(np.argmax(preds))  # type: ignore
    
    # 인덱스 범위 검사 추가
    if pred_index >= len(selected_class_names):
        print(f"⚠️ 경고: 예측 인덱스 {pred_index}가 클래스 수 {len(selected_class_names)}를 초과합니다.")
        print(f"예측 결과 형태: {preds.shape}, 클래스 이름: {selected_class_names}")
        pred_index = 0  # 안전한 기본값으로 설정
    
    pred_class = selected_class_names[pred_index]
    confidence = float(np.max(preds) * 100.0)  # type: ignore
    
    print(f"🎯 모델 원본 예측: {pred_class} (인덱스: {pred_index}, 신뢰도: {confidence:.1f}%)")
    print(f"📊 전체 모델 출력값:")
    for i, (class_name, prob) in enumerate(zip(selected_class_names, preds)):
        print(f"   {i}: {class_name}: {prob:.3f} ({prob*100:.1f}%)")
    print(f"🏷️ 사용된 클래스: {selected_class_names}")

    # 박스 기반 자동 추적 오버레이 생성 (Z-score 정규화 포함)
    boxed_base64 = None
    side_scores = {"left": 0.0, "right": 0.0}
    try:
        import cv2
        from utils.roi import summarize_side_scores, draw_boxes_on_image, calculate_roi_statistics, get_sinus_boxes, generate_gradcam_heatmap

        # ROI 통계 계산 먼저 (8클래스 모델에서만)
        gray_image = np.array(corrected_pil.convert('L'))
        boxes = get_sinus_boxes(gray_image.shape[1], gray_image.shape[0])
        
        if model_type == '8class':
            roi_stats = calculate_roi_statistics(gray_image, boxes)
            print(f"📊 ROI 통계: {roi_stats}")

            # Z-score 정규화가 적용된 스코어 계산 (ROI 통계 전달)
            side_scores: Dict[str, float] = summarize_side_scores(preds, selected_class_names, roi_stats)  # type: ignore
            print(f"🔍 ROI 스마트 재분류 결과: {side_scores}")  # 디버깅용
        else:
            # 4분류 모델: 간단한 Both 재분류 로직 적용
            print("📊 4분류 모델: 간단한 Both 재분류 로직 적용")
            
            # Both가 가장 높지만 Left나 Right와 차이가 적은 경우 재분류
            if pred_index == 3:  # Both로 예측된 경우
                both_score = float(preds[3])
                left_score = float(preds[1]) 
                right_score = float(preds[2])
                
                # Both와 Left/Right 차이가 20% 미만이면 더 높은 쪽으로 재분류
                if abs(both_score - left_score) < 0.2 or abs(both_score - right_score) < 0.2:
                    if left_score > right_score:
                        print(f"� 4분류 Both→Left 재분류: Both {both_score:.3f}, Left {left_score:.3f}")
                        pred_index = 1
                        pred_class = "Left"
                        confidence = (both_score + left_score) * 50  # 합산 신뢰도
                    else:
                        print(f"🔄 4분류 Both→Right 재분류: Both {both_score:.3f}, Right {right_score:.3f}")
                        pred_index = 2
                        pred_class = "Right" 
                        confidence = (both_score + right_score) * 50  # 합산 신뢰도
            
            side_scores = {'corrected': True if pred_index != int(np.argmax(preds)) else False}
            roi_stats = None
        
        # 🎯 핵심: ROI 재분류 결과를 최종 출력에 반영 (더 보수적 임계값 적용)
        if side_scores.get('corrected', False):  # 재분류가 적용된 경우
            left_score = side_scores.get('left', 0.0)
            right_score = side_scores.get('right', 0.0)
            both_score = side_scores.get('both', 0.0)
            normal_score = side_scores.get('normal', 0.0)
            
            # 🔍 원본 모델에서 Air fluid 점수 확인
            original_air_fluid_score = 0.0
            air_fluid_side = ""
            for i, name in enumerate(selected_class_names):
                if 'air' in name.lower() and i < len(preds):
                    score = float(preds[i])
                    if score > original_air_fluid_score:
                        original_air_fluid_score = score
                        air_fluid_side = "Left" if "left" in name.lower() else "Right"
            
            print(f"🔍 원본 Air fluid 점수: {original_air_fluid_score:.3f} ({air_fluid_side})")
            
            # Air fluid 특별 처리: 원본에서 15% 이상이면 강제 재분류
            if original_air_fluid_score > 0.15:
                print(f"💧 Air fluid 강제 재분류: {original_air_fluid_score*100:.1f}%")
                if air_fluid_side == "Left":
                    pred_class = "Left-Air fluid (Air fluid 우선 진단)"
                    confidence = max(left_score * 100, original_air_fluid_score * 100)
                else:
                    pred_class = "Right-Air fluid (Air fluid 우선 진단)"
                    confidence = max(right_score * 100, original_air_fluid_score * 100)
            else:
                # 기존 재분류 로직
                max_score = max(left_score, right_score, both_score, normal_score)
                
                # ⚠️ Normal이 여전히 높은 경우 재분류 취소 (70% 이상)
                if normal_score > 0.7:
                    print(f"⚠️ Normal 점수가 여전히 높음 ({normal_score:.3f}) - 재분류 취소")
                    pred_class = f"Normal (재분류 검토됨, {normal_score*100:.1f}%)"
                    confidence = normal_score * 100
                elif max_score == left_score and left_score > 0.3:  # 임계값 완화: 0.4 → 0.3
                    # 좌측으로 재분류 (혼탁 우선 로직)
                    if both_score > 0.2:  # 원래 Both였던 경우
                        pred_class = "Left-Haziness (재분류됨)"  # 혼탁으로 우선 분류
                        confidence = left_score * 100
                    elif normal_score > 0.2:  # 원래 Normal이었던 경우
                        pred_class = "Left-Haziness (재분류됨)"  # 혼탁으로 우선 분류
                        confidence = left_score * 100
                    else:
                        # 혼탁 우선 선택: Haziness > Air fluid > Mucosal 순서
                        haziness_found = False
                        for name in selected_class_names:
                            if name.lower().startswith('left-haziness'):
                                pred_class = name + " (혼탁 우선)"
                                confidence = left_score * 100
                                haziness_found = True
                                break
                        
                        if not haziness_found:
                            # Haziness가 없으면 Air fluid 우선
                            for name in selected_class_names:
                                if name.lower().startswith('left-air'):
                                    pred_class = name + " (스마트 강화)"
                                    confidence = left_score * 100
                                    haziness_found = True
                                    break
                        
                        if not haziness_found:
                            # 마지막으로 Mucosal
                            for name in selected_class_names:
                                if name.lower().startswith('left-'):
                                    pred_class = name + " (스마트 강화)"
                                    confidence = left_score * 100
                                    break
                
                elif max_score == right_score and right_score > 0.3:  # 임계값 완화: 0.4 → 0.3
                    # 우측으로 재분류 (혼탁 우선 로직)
                    if both_score > 0.2:  # 원래 Both였던 경우
                        pred_class = "Right-Haziness (재분류됨)"  # 혼탁으로 우선 분류
                        confidence = right_score * 100
                    elif normal_score > 0.2:  # 원래 Normal이었던 경우
                        pred_class = "Right-Haziness (재분류됨)"  # 혼탁으로 우선 분류
                        confidence = right_score * 100
                    else:
                        # 혼탁 우선 선택: Haziness > Air fluid > Mucosal 순서
                        haziness_found = False
                        for name in selected_class_names:
                            if name.lower().startswith('right-haziness'):
                                pred_class = name + " (혼탁 우선)"
                                confidence = right_score * 100
                                haziness_found = True
                                break
                        
                        if not haziness_found:
                            # Haziness가 없으면 Air fluid 우선
                            for name in selected_class_names:
                                if name.lower().startswith('right-air'):
                                    pred_class = name + " (스마트 강화)"
                                    confidence = right_score * 100
                                    haziness_found = True
                                    break
                        
                        if not haziness_found:
                            # 마지막으로 Mucosal
                            for name in selected_class_names:
                                if name.lower().startswith('right-'):
                                    pred_class = name + " (스마트 강화)"
                                    confidence = right_score * 100
                                    break
                else:
                    # 재분류 점수가 부족한 경우 원본 유지
                    print(f"ℹ️ 재분류 점수 부족 - 원본 유지: 최대점수 {max_score:.3f} < 0.3")
                    pred_class = f"{pred_class} (재분류 미적용)"
            
            print(f"✅ ROI 재분류 최종 결과: {pred_class} (신뢰도: {confidence:.1f}%)")
            print(f"   좌측: {left_score:.3f}, 우측: {right_score:.3f}, Both: {both_score:.3f}")
        else:
            print(f"ℹ️ ROI 재분류 미적용 - 원본 예측값 사용: {pred_class}")
        
        bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # GradCAM 히트맵 생성
        gradcam_heatmap = None
        active_model = None
        model_input = None
        
        try:
            # 사용할 모델 결정 (8클래스 또는 4클래스)
            active_model = model_8class if model_8class is not None else model_4class
            if active_model is not None:
                # 모델 입력용 이미지 전처리 (96x96으로 리사이즈 필수!)
                gray_resized = cv2.resize(gray_image, (96, 96))  # 96x96으로 리사이즈
                model_input = np.expand_dims(gray_resized / 255.0, axis=-1)  # 정규화 및 채널 추가
                model_input = np.expand_dims(model_input, axis=0)  # 배치 차원 추가
                
                print(f"🔍 GradCAM 생성 중... 모델 입력 형태: {model_input.shape}")
                gradcam_heatmap = generate_gradcam_heatmap(
                    model=active_model,
                    image=model_input, 
                    class_index=pred_index,
                    last_conv_layer=None  # 자동 감지
                )
                print(f"✅ GradCAM 히트맵 생성 완료: {gradcam_heatmap.shape}")
            else:
                print("⚠️ 사용할 수 있는 모델이 없어 GradCAM 생략")
                gradcam_heatmap = None
        except Exception as e:
            print(f"❌ GradCAM 생성 실패: {e}")
            gradcam_heatmap = None
        
        # Z-score가 적용된 이미지 생성 (GradCAM 포함)
        boxed_bgr = draw_boxes_on_image(
            bgr.copy(), 
            side_scores, 
            label=pred_class, 
            conf=confidence,
            gradcam_heatmap=gradcam_heatmap,  # GradCAM 히트맵 전달
            model=active_model,  # 모델 전달
            processed_image=model_input,
            pred_index=pred_index
        )  # type: ignore

        buf2 = io.BytesIO()
        img_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)  # type: ignore
        
        # 이미지 크기 최적화
        pil_img = Image.fromarray(img_rgb)  # type: ignore
        pil_img.save(buf2, format='PNG', optimize=True, compress_level=6)
        
        boxed_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        print(f"✅ ROI 이미지 생성 성공 (길이: {len(boxed_base64)})") 
        print(f"📊 ROI 이미지 크기: {len(buf2.getvalue())} bytes")  # 추가 디버깅
    except Exception as e:
        print(f"❌ ROI 이미지 생성 실패: {e}")  # 디버깅용
        import traceback
        traceback.print_exc()  # 전체 에러 스택 출력
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

@app.route('/medical')
def medical_dashboard():
    """의료진 전용 대시보드"""
    return render_template('index.html', mode='medical')

@app.route('/patient')
def patient_view():
    """환자용 화면"""
    return render_template('index.html', mode='patient')

@app.route('/api/switch-mode', methods=['POST'])
def switch_mode() -> Dict[str, str]:
    """모드 전환 API"""
    data = request.get_json()
    mode = data.get('mode', 'medical') if data else 'medical'
    return {'status': 'success', 'mode': mode}

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """AI 상담 API - 실제 진단 결과를 기반으로 응답"""
    data = request.get_json()
    question = data.get('question', '') if data else ''
    
    # 세션이나 전역변수에서 최근 진단 결과를 가져와야 하지만,
    # 간단한 구현을 위해 요청 데이터에서 가져오기
    diagnosis_data = data.get('diagnosisData', {}) if data else {} # type: ignore
    
    # 디버깅: 받은 데이터 출력
    print(f"🤖 AI 상담 요청 받음:")
    print(f"   질문: {question}")
    print(f"   진단 데이터: {diagnosis_data}")
    
    # 진단 근거 관련 질문 처리
    if any(keyword in question for keyword in ['근거', '어떤', '내 진단', 'Right-Mucosal', 'Left-Mucosal', '결과', '설명']):
        response = generate_diagnosis_explanation(question, diagnosis_data) # type: ignore
    else:
        response = generate_general_response(question)
    
    print(f"🤖 AI 응답 생성 완료 (길이: {len(response)})")
    return jsonify({'response': response})

def generate_diagnosis_explanation(question: str, diagnosis_data: Dict[str, Any]) -> str:
    """진단 근거 설명 생성"""
    prediction = diagnosis_data.get('prediction', '')
    confidence = diagnosis_data.get('confidence', 0)
    left_score = diagnosis_data.get('leftScore', 0)
    right_score = diagnosis_data.get('rightScore', 0)
    model_type = diagnosis_data.get('modelType', '8class')
    
    if not prediction:
        return "아직 진단이 수행되지 않았습니다. 먼저 X-ray 이미지를 업로드하고 분석을 진행해 주세요."
    
    response = f"**현재 진단 결과: {prediction}**\n\n"
    response += "**AI 진단 근거 상세 분석:**\n\n"
    
    # 신뢰도 설명
    if confidence > 0:
        response += f"**1. 진단 신뢰도: {confidence:.1f}%**\n"
        if confidence >= 90:
            response += "• 매우 높은 확신도로 진단되었습니다\n"
            response += "• 영상에서 명확한 특징이 관찰되었습니다\n"
        elif confidence >= 70:
            response += "• 중간 정도의 확신도로 진단되었습니다\n"
            response += "• 추가적인 임상 소견 검토가 도움이 될 수 있습니다\n"
        else:
            response += "• 비교적 낮은 확신도입니다\n"
            response += "• 재검사나 다른 진단법 고려가 필요할 수 있습니다\n"
    
    # ROI 분석 결과
    if left_score > 0 or right_score > 0:
        response += f"\n**2. 부비동 영역별 분석:**\n"
        response += f"• 좌측 부비동 이상 소견: {left_score*100:.1f}%\n"
        response += f"• 우측 부비동 이상 소견: {right_score*100:.1f}%\n"
    
    # 진단별 상세 근거
    response += "\n**3. 진단 근거 설명:**\n"
    if "Right-Mucosal" in prediction:
        response += "• **우측 상악동 점막 비후 진단:**\n"
        response += f"  - 우측 부비동에서 {right_score*100:.1f}% 확률로 이상 소견이 감지되었습니다\n"
        response += "  - 점막 비후(Mucosal thickening) 소견이 관찰됩니다\n"
        response += "  - 염증으로 인한 우측 상악동 점막의 부종이 확인됩니다\n"
        response += "  - X-ray에서 우측 상악동 부위의 혼탁도가 증가했습니다\n"
        response += "  - 정상적인 공기 음영이 감소하고 연조직 음영이 증가했습니다\n"
    elif "Left-Mucosal" in prediction:
        response += "• **좌측 상악동 점막 비후 진단:**\n"
        response += f"  - 좌측 부비동에서 {left_score*100:.1f}% 확률로 이상 소견이 감지되었습니다\n"
        response += "  - 점막 비후(Mucosal thickening) 소견이 관찰됩니다\n"
        response += "  - 염증으로 인한 좌측 상악동 점막의 부종이 확인됩니다\n"
    elif "Both" in prediction or "Bilateral" in prediction:
        response += "• **양측 부비동염 진단:**\n"
        response += f"  - 좌측 부비동 이상 소견: {left_score*100:.1f}%\n"
        response += f"  - 우측 부비동 이상 소견: {right_score*100:.1f}%\n"
        response += "  - 양쪽 부비동 모두에서 염증성 변화가 관찰됩니다\n"
        response += "  - 전반적인 부비동 염증 상태가 확인됩니다\n"
        response += "  - 좌우 대칭적 또는 비대칭적 염증 패턴을 보입니다\n"
        response += "  - X-ray에서 양측 상악동 모두 혼탁도가 증가했습니다\n"
        response += "  - 양측 모두에서 정상적인 공기 음영이 감소했습니다\n"
    elif "Normal" in prediction:
        response += "• **정상 판정 근거:**\n"
        response += "  - 양쪽 부비동 모두 정상 범위의 투명도를 보입니다\n"
        response += "  - 점막 비후나 삼출액 소견이 관찰되지 않습니다\n"
    
    # 모델 정보
    response += "\n**4. 분석 모델 정보:**\n"
    if model_type == '8class':
        response += "• 8클래스 정밀 진단 모델 사용\n"
        response += "• 좌우별, 증상별 세분화 분석 (점막비후, 기액면, 혼탁 구분)\n"
    else:
        response += "• 4클래스 빠른 진단 모델 사용\n"
    
    response += "\n**⚠️ 중요 안내:**\n"
    response += "• 본 AI 분석은 보조 진단 도구입니다\n"
    response += "• 최종 진단은 의료진의 종합적 판단이 필요합니다\n"
    
    return response

def generate_general_response(question: str) -> str:
    """일반적인 AI 응답 생성"""
    responses = {
        '부비동염': "부비동염은 부비동에 염증이 생기는 질환입니다...",
        '치료': "부비동염 치료는 항생제, 비강스프레이 등을 사용합니다...",
        '예방': "부비동염 예방을 위해서는 손씻기, 실내습도 유지 등이 중요합니다..."
    }
    
    for keyword, response in responses.items():
        if keyword in question:
            return response
    
    return "죄송합니다. 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다."

@app.route('/logout')
def logout():
    """로그아웃 처리"""
    # 실제 환경에서는 세션 관리 로직 추가
    return render_template('login.html') if False else "로그아웃 완료"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)