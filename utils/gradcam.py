import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, image_array, class_index, last_conv_layer_name='conv2d_2'):
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
    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"'{last_conv_layer_name}' 레이어가 모델에 없습니다. 현재 레이어 목록: {[l.name for l in model.layers]}")

    # 배치 차원 추가
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)

    # ✅ Grad-CAM 모델 정의
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # ✅ Gradient 계산
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]

    # ✅ Gradient → Importance score
    grads = tape.gradient(loss, conv_outputs)[0]  # shape: (H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # shape: (C,)

    conv_outputs = conv_outputs[0]  # shape: (H, W, C)
    weighted_sum = conv_outputs @ pooled_grads[..., tf.newaxis]  # shape: (H, W, 1)
    heatmap = tf.squeeze(weighted_sum)

    # ✅ 정규화 및 0~255 히트맵으로 변환
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image_array.shape[2], image_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)

    return heatmap

# ✅ 별도 함수: 마지막 Conv2D 레이어 자동 탐색
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Conv2D 레이어를 찾을 수 없습니다.")
