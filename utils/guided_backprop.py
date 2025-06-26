import tensorflow as tf
import numpy as np
import keras.backend as K

@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy
    return tf.nn.relu(x), grad

def compute_guided_backprop(model, processed_input, class_index):
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
        processed_input = tf.convert_to_tensor(processed_input, dtype=tf.float32)

    gb_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.output
    )

    # 커스텀 ReLU 적용
    for layer in gb_model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

    with tf.GradientTape() as tape:
        tape.watch(processed_input)
        preds = gb_model(processed_input)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, processed_input)
    return grads.numpy()[0]
