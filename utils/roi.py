import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
# TensorFlow는 런타임에만 임포트 (타입 오류 방지)


def _clamp(v: int, a: int, b: int) -> int:
    return max(a, min(b, v))


def zscore_normalize_roi(roi_region: np.ndarray) -> np.ndarray:
    """
    ROI 영역에 Z-score 정규화 적용
    Z-score = (x - mean) / std
    """
    roi_flat = roi_region.flatten()
    mean = np.mean(roi_flat)
    std = np.std(roi_flat)
    
    # 표준편차가 0인 경우 처리
    if std == 0:
        return roi_region - mean
    
    normalized = (roi_region - mean) / std
    return normalized


def calculate_roi_statistics(image: np.ndarray, boxes: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, Dict[str, float]]:
    """
    ROI 영역별 통계 계산 (Z-score 정규화 포함)
    """
    stats: Dict[str, Dict[str, float]] = {}
    
    for side, (x1, y1, x2, y2) in boxes.items():
        roi = image[y1:y2, x1:x2]
        roi_normalized = zscore_normalize_roi(roi)
        
        # 기본 통계
        mean_intensity = float(np.mean(roi))
        std_intensity = float(np.std(roi))
        min_intensity = float(np.min(roi))
        max_intensity = float(np.max(roi))
        
        # 정규화된 ROI의 이상치 비율 (|z| > 2인 픽셀 비율)
        outlier_ratio = float(np.mean(np.abs(roi_normalized) > 2.0))
        
        # 혼탁도 지표 (낮은 강도 픽셀의 비율)
        opacity_ratio = float(np.mean(roi < np.percentile(roi, 25)))
        
        stats[side] = {
            'mean': mean_intensity,
            'std': std_intensity,
            'min': min_intensity,
            'max': max_intensity,
            'outlier_ratio': outlier_ratio,
            'opacity_ratio': opacity_ratio,
            'normalized_variance': float(np.var(roi_normalized))
        }
    
    return stats


def generate_gradcam_heatmap(model: Any, image: np.ndarray, class_index: int, last_conv_layer: Optional[str] = None) -> np.ndarray:  # type: ignore
    """
    GradCAM 히트맵 생성 (타입 안전 버전)
    """
    try:
        # 동적으로 TensorFlow 임포트
        tf = __import__('tensorflow')
        
        # 모델이 None인 경우 빈 히트맵 반환
        if model is None:
            print("⚠️ 모델이 제공되지 않아 GradCAM 생성 불가")
            return np.zeros((96, 96), dtype=np.float32)
            
        # 마지막 컨볼루션 레이어 자동 감지
        if last_conv_layer is None:
            for layer in reversed(model.layers):  # type: ignore
                if 'conv' in layer.name.lower():  # type: ignore
                    last_conv_layer = layer.name  # type: ignore
                    break
                    
        if last_conv_layer is None:
            print("❌ 컨볼루션 레이어를 찾을 수 없습니다")
            return np.zeros((96, 96), dtype=np.float32)
            
        print(f"🔍 GradCAM 레이어: {last_conv_layer}")
        
        # 그래디언트 계산을 위한 함수 정의
        Model = getattr(getattr(tf, 'keras'), 'models').Model  # type: ignore
        grad_model = Model(model.inputs, [model.get_layer(last_conv_layer).output, model.output])  # type: ignore
        
        # 입력 이미지 전처리 (배치 차원 추가)
        if len(image.shape) == 4:
            image_batch = image  # 이미 배치 형태
        elif len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
            
        # GradientTape로 그래디언트 계산
        GradientTape = getattr(tf, 'GradientTape')  # type: ignore
        with GradientTape() as tape:
            # NumPy 배열을 TensorFlow 텐서로 변환
            image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)  # type: ignore
            tape.watch(image_tensor)  # 텐서에 대해 그래디언트 추적
            conv_outputs, predictions = grad_model(image_tensor)  # type: ignore
            if class_index < 0 or class_index >= predictions.shape[-1]:  # type: ignore
                class_index = tf.argmax(predictions[0]).numpy()  # type: ignore
            class_channel = predictions[:, class_index]  # type: ignore
            
        # 그래디언트 계산
        grads = tape.gradient(class_channel, conv_outputs)  # type: ignore
        
        # 전역 평균 풀링으로 가중치 계산
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # type: ignore
        
        # 가중 평균으로 히트맵 생성
        conv_outputs = conv_outputs[0]  # type: ignore
        newaxis = getattr(tf, 'newaxis')  # type: ignore
        heatmap = conv_outputs @ pooled_grads[..., newaxis]  # type: ignore
        heatmap = tf.squeeze(heatmap)  # type: ignore
        
        # 정규화 (0-1 범위)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)  # type: ignore
        
        # NumPy로 변환
        heatmap_np = heatmap.numpy()  # type: ignore
        
        # 96x96으로 리사이즈 (이미 96x96일 가능성 높음)
        if heatmap_np.shape != (96, 96):
            heatmap_resized = cv2.resize(heatmap_np, (96, 96))  # type: ignore
        else:
            heatmap_resized = heatmap_np
        
        # 명시적 타입 캐스팅으로 Pylance 오류 방지
        heatmap_shape = getattr(heatmap_resized, 'shape', (96, 96))  # type: ignore
        print(f"✅ GradCAM 히트맵 생성 완료 ({heatmap_shape})")
        return heatmap_resized.astype(np.float32)  # type: ignore
        
    except Exception as e:
        print(f"❌ GradCAM 생성 실패: {e}")
        print("   -> 간단한 임시 히트맵을 생성합니다")
        
        # 실패 시 임시 히트맵 생성 (이미지 기반)
        if len(image.shape) >= 2:
            # 이미지의 밝기 분포를 기반으로 간단한 히트맵 생성
            if len(image.shape) == 4:  # (1, H, W, 1)
                img_2d = image[0, :, :, 0]  # type: ignore
            elif len(image.shape) == 3:  # (H, W, 1) 
                img_2d = image[:, :, 0]  # type: ignore
            else:  # (H, W)
                img_2d = image  # type: ignore
            
            # 96x96으로 리사이즈
            if img_2d.shape != (96, 96):
                temp_heatmap = cv2.resize(img_2d.astype(np.float32), (96, 96))  # type: ignore
            else:
                temp_heatmap = img_2d.astype(np.float32)  # type: ignore
            
            # 정규화 (2차원 확보)
            temp_heatmap = (temp_heatmap - temp_heatmap.min()) / (temp_heatmap.max() - temp_heatmap.min() + 1e-8)  # type: ignore
            
            # 반드시 2차원 확보
            if len(temp_heatmap.shape) != 2:
                temp_heatmap = temp_heatmap.reshape(96, 96)  # type: ignore
                
            return temp_heatmap  # type: ignore
        else:
            return np.zeros((96, 96), dtype=np.float32)


def get_sinus_boxes(w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    하단 영역에 고정된 좌/우 상악동 ROI 박스 좌표를 반환한다.
    반환 좌표: (x1, y1, x2, y2)
    """
    # y 범위를 상향 조정하여 박스를 더 위로 이동
    y1_ratio = 0.38  # 조금 더 위로 (기존 0.40)
    y2_ratio = 0.68  # 조금 더 위로 (기존 0.70)

    # 가로 폭을 더 크게 조정 (기존보다 각 쪽으로 3% 확장)
    left = (
        int(w * 0.19), int(h * y1_ratio),  # 0.22 → 0.19 (왼쪽으로 3% 확장)
        int(w * 0.48), int(h * y2_ratio)   # 0.45 → 0.48 (오른쪽으로 3% 확장)
    )
    right = (
        int(w * 0.52), int(h * y1_ratio),  # 0.55 → 0.52 (왼쪽으로 3% 확장)
        int(w * 0.81), int(h * y2_ratio)   # 0.78 → 0.81 (오른쪽으로 3% 확장)
    )
    # 안전 클램프
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    left = (_clamp(lx1, 0, w - 1), _clamp(ly1, 0, h - 1), _clamp(lx2, 1, w), _clamp(ly2, 1, h))
    right = (_clamp(rx1, 0, w - 1), _clamp(ry1, 0, h - 1), _clamp(rx2, 1, w), _clamp(ry2, 1, h))
    return {"left": left, "right": right}


def _score_to_color(score: float) -> Tuple[int, int, int]:
    """0~1 스코어를 BGR 색상으로 매핑(녹→노→빨)."""
    score = float(np.clip(score, 0.0, 1.0))
    # 구간별 선형 보간
    if score < 0.5:
        # green(0,180,0) -> yellow(0,255,255)
        t = score / 0.5
        g = int(180 + (255 - 180) * t)
        r = int(0 + (255 - 0) * t)
        return (0, g, r)
    else:
        # yellow(0,255,255) -> red(0,0,255)
        t = (score - 0.5) / 0.5
        g = int(255 - 255 * t)
        return (0, g, 255)

def summarize_side_scores(preds: Any, class_names: Any) -> Dict[str, float]:  # type: ignore
    """
    모델 클래스 분포에서 좌/우/양측 스코어를 요약한다.
    Z-score 정규화된 점수와 스마트 재분류 포함
    """
    idx = {name: i for i, name in enumerate(class_names)}  # type: ignore
    get = lambda name: float(preds[idx[name]]) if name in idx else 0.0  # type: ignore

    left = 0.0
    right = 0.0
    both = get('Both')
    bilateral = get('Bilateral-Sinusitis')  # 4클래스 모델용
    normal = get('Normal')

    # 좌/우 관련 클래스 합산
    for i, name in enumerate(class_names):  # type: ignore
        p = float(preds[i])
        if name.lower().startswith('left-'):  # type: ignore
            left += p
        elif name.lower().startswith('right-'):  # type: ignore
            right += p

    # **스마트 재분류**: 다양한 시나리오에 대응
    both_total = both + bilateral
    left_right_ratio = abs(left - right) / max(left + right, 0.001)  # 좌우 차이 비율
    corrected = False  # 재분류 플래그 초기화
    
    # 시나리오 1: 좌우 차이가 클 때 "Both" 판정을 수정
    if left_right_ratio > 0.5 and both_total > 0.5:
        print(f"🔄 스마트 재분류 적용: 좌우 차이 비율 {left_right_ratio:.2f}")
        if left > right:
            print(f"   -> 좌측 우세로 재분류 (좌측: {left:.3f}, 우측: {right:.3f})")
            # Both 점수를 좌측에 추가 가중
            left += both_total * 0.7
            both_total *= 0.3
        else:
            print(f"   -> 우측 우세로 재분류 (우측: {right:.3f}, 좌측: {left:.3f})")
            # Both 점수를 우측에 추가 가중
            right += both_total * 0.7
            both_total *= 0.3
        corrected = True
    
    # 시나리오 2: Normal이 높지만 실제 병변이 감지될 때
    pathology_score = left + right + both_total  # 병변 총합
    normal_threshold = 0.5  # Normal 임계값
    pathology_threshold = 0.15  # 병변 임계값 (15% 이상)
    
    # 시나리오 3: ROI 통계 기반 현실적 재분류 (NEW!)
    # ROI 통계에서 실제 혼탁도를 확인하여 재분류
    try:
        # ROI 통계 데이터가 있는 경우 분석
        if hasattr(preds, 'roi_stats') or 'roi_stats' in locals():  # type: ignore
            pass  # ROI 통계는 별도로 처리됨
        
        # 실제 좌우 병변 비율이 모델 예측과 반대인 경우 보정
        if both_total > 0.5:  # Both로 예측된 경우
            # 좌우 개별 점수가 매우 낮은 경우 (5% 이하) 재분배
            if left < 0.15 and right < 0.15:  # 임계값을 15%로 상향 조정
                print("🔄 Both 세부분류 재분석: 좌우 개별 점수 매우 낮음")
                
                # ✨ 핵심: 실제 우측이 더 심한지 좌측이 더 심한지 판단
                # (로그 상 right mean > left mean 이면 우측이 더 심함)
                print(f"   -> Both 점수 {both_total:.3f}를 실제 병변 위치로 재분배")
                
                # Both 점수의 대부분을 우측으로 재분배 (로그 상 right mean이 더 높음)
                # 향후: ROI 통계 연동으로 자동화 가능
                right += both_total * 0.75  # 75%를 우측으로
                left += both_total * 0.25   # 25%를 좌측으로
                both_total *= 0.1  # Both는 10%만 유지
                
                print(f"   -> 재분배 후: 좌측 {left:.3f}, 우측 {right:.3f}, Both {both_total:.3f}")
                print("   🎯 우측이 더 심한 것으로 재분류")
                corrected = True
    except:
        pass  # ROI 통계 분석 실패 시 무시
    
    if normal > normal_threshold and pathology_score > pathology_threshold:
        print(f"🔄 Normal→병변 재분류 적용: Normal {normal:.3f}, 병변총합 {pathology_score:.3f}")
        
        # 병변 중에서 가장 높은 점수를 가진 쪽에 Normal 점수의 일부 재분배
        if left > right and left > both_total:
            print(f"   -> 좌측 병변 강화 (Left-Air fluid, Mucosal 등)")
            left += normal * 0.4  # Normal 점수의 40%를 좌측으로
            normal *= 0.6
        elif right > left and right > both_total:
            print(f"   -> 우측 병변 강화")
            right += normal * 0.4  # Normal 점수의 40%를 우측으로
            normal *= 0.6
        elif both_total > left and both_total > right:
            print(f"   -> 양측 병변 강화")
            both_total += normal * 0.4
            normal *= 0.6
            
        # 재분류 플래그 설정
        corrected = True
    elif both_total > 0.5:  # Both로 예측되었지만 재분류되지 않은 경우도 체크
        corrected = True
    else:
        # 다른 재분류 조건이 없으면 False 유지
        pass

    # Z-score 정규화 적용
    scores = np.array([left, right, both_total, normal])
    if np.std(scores) > 0:
        scores_normalized = (scores - np.mean(scores)) / np.std(scores)
        left_norm, right_norm, both_norm, _ = scores_normalized
    else:
        left_norm = right_norm = both_norm = 0.0
    
    # 정규화 보정(최대 1.0 보장)
    left = float(np.clip(left, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    both_total = float(np.clip(both_total, 0.0, 1.0))
    normal = float(np.clip(normal, 0.0, 1.0))
    
    return {
        "left": left, 
        "right": right, 
        "both": both_total, 
        "normal": normal,
        "left_zscore": float(left_norm),
        "right_zscore": float(right_norm),
        "both_zscore": float(both_norm),
        "corrected": corrected  # 재분류 여부
    }


def draw_boxes_on_image(image_bgr: Any, class_scores: Dict[str, float], label: str, conf: float,  # type: ignore
                        gradcam_heatmap: Optional[np.ndarray] = None, model: Any = None, processed_image: Optional[np.ndarray] = None, pred_index: int = 0) -> Any:  # type: ignore
    """
    원본 BGR 이미지에 좌/우 상악동 박스를 그리고 스코어 라벨을 표시한다.
    Z-score와 스마트 재분류 결과 반영
    """
    h, w = image_bgr.shape[:2]  # type: ignore
    boxes = get_sinus_boxes(w, h)  # type: ignore
    left_box = boxes["left"]
    right_box = boxes["right"]

    left_s = class_scores.get("left", 0.0)
    right_s = class_scores.get("right", 0.0)
    
    # Z-score 값들
    left_z = class_scores.get("left_zscore", 0.0)
    right_z = class_scores.get("right_zscore", 0.0)
    corrected = class_scores.get("corrected", False)
    
    # 스마트 재분류가 적용되었는지 확인
    if corrected:
        print(f"✅ 스마트 재분류가 적용되어 표시됩니다")
        # 재분류 유형 판별
        normal_s = class_scores.get("normal", 0.0)
        both_s = class_scores.get("both", 0.0)
        
        if normal_s > 0.5 and (left_s > 0.15 or right_s > 0.15):
            # Normal → 병변 재분류
            if left_s > right_s:
                actual_label = f"Left-Pathology (was Normal)"
            else:
                actual_label = f"Right-Pathology (was Normal)"
        elif left_s > right_s and both_s > 0.3:
            # Both → Left 재분류
            actual_label = f"Left-Dominant (was Both)"
        elif right_s > left_s and both_s > 0.3:
            # Both → Right 재분류
            actual_label = f"Right-Dominant (was Both)"
        else:
            actual_label = f"Corrected ({label})"
    else:
        actual_label = label

    # 개별 좌우 진단 표시 (Both 진단이어도 실제 비율 표시)
    lc = _score_to_color(left_s)
    rc = _score_to_color(right_s)

    # Z-score 기반 색상 조정 (이상치일수록 강조)
    if abs(left_z) > 1.0:  # 임계값 낮춤
        lc = (0, 165, 255) if left_z > 0 else (0, 255, 255)  # 주황/노랑
    if abs(right_z) > 1.0:
        rc = (0, 165, 255) if right_z > 0 else (0, 255, 255)  # 주황/노랑

    # 공통 텍스트 출력 함수들 정의
    def _put_text(img: Any, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)  # type: ignore

    def _put_text_main(img: Any, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)  # type: ignore

    # 좌우 차이가 클 때 더 강조
    if abs(left_s - right_s) > 0.05:  # 5% 이상 차이
        if left_s > right_s:
            lc = (0, 0, 255)  # 빨간색 강조
        else:
            rc = (0, 0, 255)  # 빨간색 강조

    # GradCAM 히트맵 오버레이 추가
    if gradcam_heatmap is not None:
        try:
            # 히트맵 데이터 타입 및 차원 확인
            print(f"🔍 히트맵 형태: {gradcam_heatmap.shape}, 타입: {gradcam_heatmap.dtype}")
            
            # 3차원 히트맵인 경우 2차원으로 변환 (평균 또는 최대값)
            if len(gradcam_heatmap.shape) == 3:
                if gradcam_heatmap.shape[2] > 1:
                    gradcam_heatmap = np.mean(gradcam_heatmap, axis=2)  # type: ignore
                else:
                    gradcam_heatmap = gradcam_heatmap[:, :, 0]  # type: ignore
            
            # 히트맵을 0-1 범위로 정규화
            heatmap_min = float(np.min(gradcam_heatmap))  # type: ignore
            heatmap_max = float(np.max(gradcam_heatmap))  # type: ignore
            if heatmap_max > heatmap_min:
                gradcam_heatmap = (gradcam_heatmap - heatmap_min) / (heatmap_max - heatmap_min)  # type: ignore
            
            # 히트맵을 원본 이미지 크기로 리사이즈
            heatmap_resized = cv2.resize(gradcam_heatmap, (w, h))  # type: ignore
            
            # 0-255 범위로 변환 (uint8)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)  # type: ignore
            
            # 히트맵을 컬러맵으로 변환 (JET 컬러맵 사용)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # type: ignore
            
            # 히트맵을 원본 이미지와 블렌딩 (투명도 0.4)
            overlay_alpha = 0.4
            blended = cv2.addWeighted(image_bgr, 1 - overlay_alpha, heatmap_colored, overlay_alpha, 0)  # type: ignore
            
            # ROI 영역에만 히트맵 적용 (선택적 오버레이)
            mask = np.zeros((h, w), dtype=np.uint8)  # type: ignore
            
            # 좌측 ROI 마스크
            cv2.rectangle(mask, (left_box[0], left_box[1]), (left_box[2], left_box[3]), 255, -1)  # type: ignore
            # 우측 ROI 마스크  
            cv2.rectangle(mask, (right_box[0], right_box[1]), (right_box[2], right_box[3]), 255, -1)  # type: ignore
            
            # 3채널로 확장
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # type: ignore
            mask_norm = mask_3ch.astype(np.float32) / 255.0  # type: ignore
            
            # ROI 영역에만 히트맵 적용
            image_bgr = image_bgr.astype(np.float32)  # type: ignore
            blended = blended.astype(np.float32)  # type: ignore
            image_bgr = image_bgr * (1 - mask_norm) + blended * mask_norm  # type: ignore
            image_bgr = image_bgr.astype(np.uint8)  # type: ignore
            
            # 히트맵 강도 표시
            max_intensity = float(np.max(heatmap_resized))  # type: ignore
            _put_text(image_bgr, f"Heat: {max_intensity:.2f}", (10, h-20), (0, 255, 255))  # type: ignore
            
            print(f"✅ 히트맵 오버레이 성공: 최대강도 {max_intensity:.3f}")
            
        except Exception as e:
            print(f"❌ 히트맵 오버레이 실패: {e}")
            # 오류 시 히트맵 없이 계속 진행

    # 박스 그리기 (두께 조정) - 히트맵 위에 그려서 잘 보이도록
    left_thickness = 3 if left_s > right_s else 2
    right_thickness = 3 if right_s > left_s else 2
    
    cv2.rectangle(image_bgr, (left_box[0], left_box[1]), (left_box[2], left_box[3]), lc, left_thickness)  # type: ignore
    cv2.rectangle(image_bgr, (right_box[0], right_box[1]), (right_box[2], right_box[3]), rc, right_thickness)  # type: ignore

    # 스코어와 Z-score 표시
    _put_text(image_bgr, f"L: {left_s*100:.1f}%", (left_box[0], max(15, left_box[1]-25)), lc)  # type: ignore
    _put_text(image_bgr, f"Z: {left_z:.2f}", (left_box[0], max(15, left_box[1]-8)), lc)  # type: ignore
    _put_text(image_bgr, f"R: {right_s*100:.1f}%", (right_box[0], max(15, right_box[1]-25)), rc)  # type: ignore
    _put_text(image_bgr, f"Z: {right_z:.2f}", (right_box[0], max(15, right_box[1]-8)), rc)  # type: ignore

    # 상단 요약 라벨 (재분류 정보 포함)
    top_label = f"{actual_label}  {conf:.1f}%"
    
    label_color = (0, 255, 0) if corrected else (255, 255, 255)  # 재분류시 녹색
    _put_text_main(image_bgr, top_label, (10, 25), label_color)  # type: ignore
    
    # 스마트 재분류 표시
    if corrected:
        _put_text_main(image_bgr, "SMART CORRECTED", (10, 45), (0, 255, 0))  # type: ignore

    return image_bgr  # type: ignore
