import cv2
import numpy as np
import tensorflow as tf  # type: ignore
from typing import Any

def generate_gradcam(model: Any, image_array: Any, class_index: int, last_conv_layer_name: str = 'conv2d_2') -> Any:  # type: ignore
    """
    Grad-CAM 히트맵 생성 함수

    Args:
        model: 학습된 모델
        image_array: (H, W, C) 또는 (1, H, W, C) 형태의 NumPy 배열
        class_index: 예측 클래스 인덱스
        last_conv_layer_name: 마지막 Conv 레이어 이름

    Returns:
        heatmap: OpenCV 히트맵 (uint8)
    """
    # ✅ 마지막 Conv 레이어가 존재하는지 확인
    if last_conv_layer_name not in [layer.name for layer in model.layers]:  # type: ignore
        raise ValueError(f"'{last_conv_layer_name}' 레이어가 모델에 없습니다. 현재 레이어 목록: {[l.name for l in model.layers]}")  # type: ignore

    # 배치 차원 추가
    if image_array.ndim == 3:  # type: ignore
        image_array = np.expand_dims(image_array, axis=0)  # type: ignore

    # ✅ Grad-CAM 모델 정의
    # ⚠️ 입력 구조 불일치 경고 방지: 단일 입력 모델은 model.input 사용
    grad_model = tf.keras.models.Model(  # type: ignore
        inputs=model.input,  # type: ignore
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]  # type: ignore
    )

    # ✅ Gradient 계산
    with tf.GradientTape() as tape:  # type: ignore
        conv_outputs, predictions = grad_model(image_array)  # type: ignore
        # 다중 출력 모델 대응: 첫 번째 로짓 텐서 선택
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]  # type: ignore
        loss = predictions[:, class_index]  # type: ignore

    # ✅ Gradient → Importance score
    grads = tape.gradient(loss, conv_outputs)[0]  # type: ignore
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # type: ignore

    conv_outputs = conv_outputs[0]  # type: ignore
    weighted_sum = conv_outputs @ pooled_grads[..., tf.newaxis]  # type: ignore
    heatmap = tf.squeeze(weighted_sum)  # type: ignore

    # ✅ 정규화 및 0~255 히트맵으로 변환
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)  # type: ignore
    heatmap = heatmap.numpy()  # type: ignore
    heatmap = cv2.resize(heatmap, (image_array.shape[2], image_array.shape[1]))  # type: ignore
    heatmap = np.uint8(255 * heatmap)  # type: ignore

    return heatmap

# ✅ 별도 함수: 마지막 Conv2D 레이어 자동 탐색
def find_last_conv_layer(model: Any) -> str:  # type: ignore
    for layer in reversed(model.layers):  # type: ignore
        if isinstance(layer, tf.keras.layers.Conv2D):  # type: ignore
            return layer.name  # type: ignore
    raise ValueError("Conv2D 레이어를 찾을 수 없습니다.")
