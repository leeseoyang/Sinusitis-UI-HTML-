import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from .guided_backprop import compute_guided_backprop
from .gradcam import generate_gradcam, find_last_conv_layer  # 🔥 추가

def guided_gradcam(model, processed_input, class_index, last_conv_layer_name=None):
    """
    Guided Grad-CAM을 생성하고 시각화 이미지(base64)로 반환

    Args:
        model: 학습된 keras 모델
        processed_input: (1, H, W, C) 형태의 입력 이미지
        class_index: 예측된 클래스 인덱스
        last_conv_layer_name: 마지막 Conv 레이어 이름 (None이면 자동탐색)

    Returns:
        guided_gradcam_output: np.array (가이드드 그래드캠 결과 배열)
        encoded: base64 인코딩된 이미지
    """
    # ✅ Conv 레이어 자동 선택
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # 배치 차원 보장
    if processed_input.ndim == 3:
        processed_input = np.expand_dims(processed_input, axis=0)

    # Guided Backprop 결과
    gb = compute_guided_backprop(model, processed_input, class_index)

    # Grad-CAM 히트맵
    cam = generate_gradcam(model, processed_input[0], class_index, last_conv_layer_name)
    cam = cam.astype(np.float32) / 255.0
    cam = np.expand_dims(cam, axis=-1)

    # 곱하여 guided grad-cam 생성
    guided_gradcam_output = gb * cam

    input_img = processed_input[0]
    gradcam_img = cam.squeeze()
    guided_img = np.sum(guided_gradcam_output, axis=-1)

    # base64 시각화 처리
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(input_img)
    axs[0].set_title("입력 이미지")
    axs[0].axis('off')
    axs[1].imshow(gradcam_img, cmap='jet')
    axs[1].set_title("Grad-CAM")
    axs[1].axis('off')
    axs[2].imshow(guided_img, cmap='gray')
    axs[2].set_title("Guided Grad-CAM")
    axs[2].axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return guided_gradcam_output, encoded
