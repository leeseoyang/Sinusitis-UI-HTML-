import cv2
import numpy as np
from typing import Tuple, Dict, Any


def _clamp(v: int, a: int, b: int) -> int:
    return max(a, min(b, v))


def get_sinus_boxes(w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    하단 영역에 고정된 좌/우 상악동 ROI 박스 좌표를 반환한다.
    반환 좌표: (x1, y1, x2, y2)
    """
    # y 범위를 상향 조정하여 박스를 더 위로 이동
    y1_ratio = 0.38  # 조금 더 위로 (기존 0.40)
    y2_ratio = 0.68  # 조금 더 위로 (기존 0.70)

    left = (
        int(w * 0.22), int(h * y1_ratio),
        int(w * 0.45), int(h * y2_ratio)
    )
    right = (
        int(w * 0.55), int(h * y1_ratio),
        int(w * 0.78), int(h * y2_ratio)
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
    클래스 네이밍 규칙에 의존: 'Left-', 'Right-', 'Both', 'Normal'
    반환: {left:0~1, right:0~1, both:0~1, normal:0~1}
    """
    idx = {name: i for i, name in enumerate(class_names)}  # type: ignore
    get = lambda name: float(preds[idx[name]]) if name in idx else 0.0  # type: ignore

    left = 0.0
    right = 0.0
    both = get('Both')
    normal = get('Normal')

    # 좌/우 관련 클래스 합산
    for i, name in enumerate(class_names):  # type: ignore
        p = float(preds[i])
        if name.lower().startswith('left-'):  # type: ignore
            left += p
        elif name.lower().startswith('right-'):  # type: ignore
            right += p

    # 양측 가중 분배(선택): both가 높을수록 좌/우에 반영
    left += both * 0.5
    right += both * 0.5

    # 정규화 보정(최대 1.0 보장)
    left = float(np.clip(left, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    both = float(np.clip(both, 0.0, 1.0))
    normal = float(np.clip(normal, 0.0, 1.0))
    return {"left": left, "right": right, "both": both, "normal": normal}


def draw_boxes_on_image(image_bgr: Any, class_scores: Dict[str, float],  # type: ignore
                        label: str, conf: float) -> Any:  # type: ignore
    """
    원본 BGR 이미지에 좌/우 상악동 박스를 그리고 스코어 라벨을 표시한다.
    - class_scores: summarize_side_scores 결과
    - label/conf: 최종 예측 요약
    반환: BGR 이미지
    """
    h, w = image_bgr.shape[:2]  # type: ignore
    boxes = get_sinus_boxes(w, h)  # type: ignore
    left_box = boxes["left"]
    right_box = boxes["right"]

    left_s = class_scores.get("left", 0.0)
    right_s = class_scores.get("right", 0.0)

    # 색상 결정
    lc = _score_to_color(left_s)
    rc = _score_to_color(right_s)

    # 박스 그리기
    cv2.rectangle(image_bgr, (left_box[0], left_box[1]), (left_box[2], left_box[3]), lc, 2)  # type: ignore
    cv2.rectangle(image_bgr, (right_box[0], right_box[1]), (right_box[2], right_box[3]), rc, 2)  # type: ignore

    # 라벨 텍스트
    def _put_text(img: Any, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)  # type: ignore
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)  # type: ignore

    _put_text(image_bgr, f"L: {left_s*100:.1f}%", (left_box[0], max(15, left_box[1]-8)), lc)  # type: ignore
    _put_text(image_bgr, f"R: {right_s*100:.1f}%", (right_box[0], max(15, right_box[1]-8)), rc)  # type: ignore

    # 상단 요약 라벨
    top_label = f"{label}  {conf:.1f}%"
    _put_text(image_bgr, top_label, (10, 20), (255, 255, 255))  # type: ignore

    return image_bgr  # type: ignore
