import tensorflow as tf
# TFRecord 경로
tfrecord_path = r"C:\Users\winte\Desktop\jjm\1. 업무\CNN_ECG_Event\ecg_dataset.tfrecord"
# feature schema 정의
feature_description = {
    "ecg": tf.io.FixedLenFeature([250], tf.float32),
    "label": tf.io.FixedLenFeature([250], tf.int64),
    "subject_id": tf.io.FixedLenFeature([], tf.int64),
}
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)
# TFRecordDataset 로딩 및 파싱
dataset = tf.data.TFRecordDataset(tfrecord_path)
parsed_dataset = dataset.map(_parse_function)
# subject_id = 0 이고, label에 VPC (2)가 있는 샘플 카운트
vpc_count = 0
sample_count = 0
for example in parsed_dataset:
    if example["subject_id"].numpy() == 0:
        label = example["label"].numpy()
        if 2 in label:
            vpc_count += 1
        sample_count += 1
print(f"\n Subject ID 0:")
print(f" 전체 샘플 수: {sample_count}")
print(f" VPC 라벨이 포함된 샘플 수: {vpc_count}")