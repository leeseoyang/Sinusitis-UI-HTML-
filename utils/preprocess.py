import numpy as np
import cv2
from PIL import Image
import pytesseract

def preprocess_image(image, target_size=(96, 96)):
    """
    이미지 전처리: PIL.Image 또는 NumPy 배열을 받아서 정규화된 NumPy 배열 반환
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("이미지는 3채널(RGB)이어야 합니다.")

    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image.astype(np.float32)


def correct_flip(image_np):
    """
    좌우 반전 여부 판단: OCR + 밝기 분석
    Returns:
        flip_flag: str ("Flip Applied" or "No Flip")
        corrected_pil: PIL.Image
    """
    pil_image = Image.fromarray(image_np)
    try:
        ocr_text = pytesseract.image_to_string(pil_image)
        print(f"[DEBUG OCR 결과] >>> {ocr_text.strip()}")
    except Exception as e:
        print(f"[WARNING] OCR 오류 발생: {e}")
        ocr_text = ""

    if 'L' in ocr_text.upper() or 'LEFT' in ocr_text.upper():
        return "Flip Applied", pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    if 'R' in ocr_text.upper() or 'RIGHT' in ocr_text.upper():
        return "No Flip", pil_image

    # OCR 실패 시 밝기 기준 좌우 판단
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    roi = gray[0:h//2, 0:w//2]
    bright_ratio = np.sum(roi > 200) / roi.size
    print(f"[DEBUG 픽셀 밝기비율] 좌상단 25% 밝은 픽셀 비율: {bright_ratio:.2f}")

    if bright_ratio > 0.1:
        return "No Flip", pil_image
    else:
        return "Flip Applied", pil_image.transpose(Image.FLIP_LEFT_RIGHT)


def preprocess_and_correct(image, target_size=(96, 96)):
    """
    단일 이미지 전처리 + 방향 보정
    Returns:
        processed_np: 모델 입력용 NumPy 배열
        flip_flag: str
        corrected_pil: PIL.Image
    """
    if not isinstance(image, Image.Image):
        raise TypeError("입력은 반드시 PIL.Image 객체여야 합니다.")

    image_np = np.array(image)
    flip_flag, corrected_pil = correct_flip(image_np)
    processed_np = preprocess_image(corrected_pil, target_size)
    return processed_np, flip_flag, corrected_pil


def preprocess_and_correct_batch(images, target_size=(96, 96)):
    """
    여러 이미지 전처리 + 방향 보정 (배치 처리용)
    """
    processed_images = []
    flip_flags = []
    corrected_images = []

    for image in images:
        processed_np, flip_flag, corrected_pil = preprocess_and_correct(image, target_size)
        processed_images.append(processed_np)
        flip_flags.append(flip_flag)
        corrected_images.append(corrected_pil)

    return np.array(processed_images), flip_flags, corrected_images
