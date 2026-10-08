import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b1 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b1)
    merge1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge1)

    u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    merge2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge2)

    u3 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
    merge3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge3)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c6)
    return models.Model(inputs=[inputs], outputs=[outputs])

def calculate_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def test_unet_output_shape():
    """Verify that U-Net produces binary mask probability map matching input spatial dimensions."""
    model = build_unet(input_shape=(128, 128, 3))
    dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    output = model.predict(dummy_input)
    assert output.shape == (1, 128, 128, 1), f"Expected shape (1, 128, 128, 1), got {output.shape}"
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "Outputs must be valid probabilities in [0, 1]"

def test_dice_coefficient_metric():
    """Verify Dice metric calculation: perfect match must yield ~1.0, complete mismatch must yield ~0.0."""
    mask_a = np.ones((128, 128, 1), dtype=np.float32)
    mask_b = np.ones((128, 128, 1), dtype=np.float32)
    perfect_dice = calculate_dice(mask_a, mask_b)
    assert np.isclose(perfect_dice, 1.0, atol=1e-4), f"Expected Dice ~1.0, got {perfect_dice}"

    mask_zeros = np.zeros((128, 128, 1), dtype=np.float32)
    zero_dice = calculate_dice(mask_a, mask_zeros)
    assert np.isclose(zero_dice, 0.0, atol=1e-4), f"Expected Dice ~0.0, got {zero_dice}"
