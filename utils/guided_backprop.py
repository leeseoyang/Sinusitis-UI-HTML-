from typing import Tuple, Callable  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import activations  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import numpy as np
from typing import Any

@tf.custom_gradient  # type: ignore
def guided_relu(x):  # type: ignore
    def grad(dy):  # type: ignore
        return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy  # type: ignore
    return activations.relu(x), grad  # type: ignore

def compute_guided_backprop(model: Any, processed_input: Any, class_index: int) -> Any:  # type: ignore
    """
    모델의 입력에 대해 Guided Backpropagation을 수행한다.
    
    Args:
        model: 학습된 keras 모델
        processed_input: (1, H, W, C) numpy array
        class_index: 타깃 클래스 인덱스
    
    Returns:
        guided_grad: numpy array
    """
    # 💥 NumPy → Tensor 변환
    if isinstance(processed_input, np.ndarray):
        processed_input = tf.convert_to_tensor(processed_input)  # type: ignore
        
    # ⚠️ 입력 구조 불일치 방지: 단일 입력 모델은 model.input 사용
    gb_model = Model(inputs=model.inputs, outputs=model.outputs)  # type: ignore

    # 커스텀 ReLU 적용
    for layer in gb_model.layers:  # type: ignore
        if hasattr(layer, 'activation') and layer.activation == activations.relu:  # type: ignore
            layer.activation = guided_relu  # type: ignore

    with tf.GradientTape() as tape:  # type: ignore
        tape.watch(processed_input)  # type: ignore
        preds = gb_model(processed_input)  # type: ignore
        loss = preds[:, class_index]  # type: ignore

    grads = tape.gradient(loss, processed_input)  # type: ignore
    if grads is None:  # type: ignore
        return np.zeros_like(processed_input.numpy()[0])  # type: ignore
    
    # tape.gradient can return a list for multi-input models, handle this case.
    if isinstance(grads, list):  # type: ignore
        grads = grads[0]  # type: ignore

    if grads is None:  # type: ignore
        return np.zeros_like(processed_input.numpy()[0])  # type: ignore
        
    return grads.numpy()[0]  # type: ignore
