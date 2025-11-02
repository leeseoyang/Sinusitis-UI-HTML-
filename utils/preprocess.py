from PIL import Image
import numpy as np

IMAGE_SIZE = (96, 96)


def _to_grayscale(pil_image: Image.Image) -> Image.Image:
    """모델 학습과 동일하게 단일 채널(R=G=B)로 맞춘다."""
    return pil_image.convert('L')


def _normalize(image_array: np.ndarray) -> np.ndarray:
    """0~255 → 0~1 부동소수 정규화."""
    return image_array.astype(np.float32) / 255.0


def preprocess_and_correct(pil_image):
    """
    학습 시 사용한 파이프라인과 동일한 방식으로 전처리.
    Args:
        pil_image: PIL Image 객체
    Returns:
        tuple: (모델 입력용 numpy array(H,W,1), 표시용 PIL 이미지)
    """
    gray = _to_grayscale(pil_image)
    resized = gray.resize(IMAGE_SIZE)
    processed = np.array(resized, dtype=np.float32)
    processed = _normalize(processed)
    processed = np.expand_dims(processed, axis=-1)

    corrected_pil = pil_image.copy()
    return processed, corrected_pil


def preprocess_xray(image_np):
    """
    배열 형태의 이미지를 모델 입력 형태로 변환한다.
    """
    if len(image_np.shape) == 3:
        pil_image = Image.fromarray(image_np).convert('L')
    else:
        pil_image = Image.fromarray(image_np.astype(np.uint8))

    resized = pil_image.resize(IMAGE_SIZE)
    processed = np.array(resized, dtype=np.float32)
    processed = _normalize(processed)
    processed = np.expand_dims(processed, axis=-1)

    return processed
