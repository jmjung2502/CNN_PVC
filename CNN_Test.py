import os
import tensorflow as tf

# ---- TFRecord 파일 경로 ----
tfrecord_path = r"C:\Users\winte\Desktop\jjm\1. 업무\CNN_ECG_Event\ecg_dataset.tfrecord"
# ----------------------------

feature_description = {
    "ecg": tf.io.FixedLenFeature([250], tf.float32),
    "label": tf.io.FixedLenFeature([250], tf.int64),
    "subject_id": tf.io.FixedLenFeature([], tf.int64),
    "filename": tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.expand_dims(parsed['ecg'], -1)     # (250, 1)
    label = parsed['label']                     # (250,)
    return ecg, label  # filename은 학습에선 생략

def load_dataset(tfrecord_path, batch_size=64, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_1d_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(250, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'),
        tf.keras.layers.UpSampling1D(size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling1D(size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),

        # 출력 길이 조정: (252 → 250)
        tf.keras.layers.Cropping1D(cropping=(1, 1)),

        tf.keras.layers.Conv1D(8, kernel_size=1, activation='softmax', padding='same')  # (250, 8)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    dataset = load_dataset(tfrecord_path, batch_size=64)
    model = build_1d_cnn_model()
    model.summary()
    model.fit(dataset, epochs=10)

