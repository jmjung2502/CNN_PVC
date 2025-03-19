import os
import numpy as np
import tensorflow as tf

# -------- 설정 --------
BASE_PATH = r"C:\CCO_Clinic_Data\04. 고신대 성인홀터임상(임성일교수님)\01. HiCardi+ data\03. parsing data(IRB승인 대상자 100명, 2024-08-08)"
# BASE_PATH = r"C:\CCO_Clinic_Data\04. 고신대 성인홀터임상(임성일교수님)\01. HiCardi+ data\03. parsing data(IRB승인 대상자 100명, 2024-08-08)\test"
PACKET_SIZE = 156
ECG_INFO_INDEX = 11
ECG_WAVEFORM_INDEX = 14
WINDOW_SIZE = 5
STRIDE = 1
save_dir = r"C:\Users\winte\Desktop\jjm\1. 업무\CNN_ECG_Event"
tfrecord_path = os.path.join(save_dir, "ecg_dataset.tfrecord")
# ---------------------

PEAK_INFO_MAP = {
    "NaN": 0, "R peak (〮)": 1, "VPC (V)": 2, "APC (A)": 3,
    "Block beat (B)": 4, "Paced beat (P)": 5, "Pause": 6, "Unknown (Q)": 7
}

def parse_ecg_info(ecg_data):
    p_info = ecg_data[0]
    p1 = ecg_data[1]
    p2 = ecg_data[2]
    p2_peak_info = (p_info >> 4) & 0b111
    p1_peak_info = p_info & 0b111
    asys_flag = (p1 >> 6) & 0b1
    peak_index1 = p1 & 0b111111
    rpo_flag = (p2 >> 7) & 0b1
    noise_flag = (p2 >> 6) & 0b1
    peak_index2 = p2 & 0b111111
    return {
        "P1 Peak Info": PEAK_INFO_MAP.get(p1_peak_info),
        "P1 Peak Label": p1_peak_info,
        "P2 Peak Info": PEAK_INFO_MAP.get(p2_peak_info),
        "P2 Peak Label": p2_peak_info,
        "P1": {"Asystole": bool(asys_flag), "Peak Index": peak_index1},
        "P2": {"R position shift": bool(rpo_flag), "Noise": bool(noise_flag), "Peak Index": peak_index2}
    }

def convert_ecg_waveform(ecg_bytes):
    ecg_waveform = []
    for i in range(0, len(ecg_bytes), 2):
        xECG = (ecg_bytes[i] * 256) + ecg_bytes[i + 1]
        ecg_waveform.append(xECG - 32500)
    return ecg_waveform

def serialize_example(ecg_waveform, label, subject_id, filename):
    feature = {
        "ecg": tf.train.Feature(float_list=tf.train.FloatList(value=ecg_waveform)),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        "subject_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[subject_id])),
        "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

# TFRecordWriter는 단 한 번만 열고 사용
writer = tf.io.TFRecordWriter(tfrecord_path)
packet_buffer = []

for subject_id, folder in enumerate(sorted(os.listdir(BASE_PATH))):
    folder_path = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith("_raw.txt"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            with open(file_path, "r") as f:
                for line in f:
                    values = list(map(int, line.strip().split()))
                    if len(values) != PACKET_SIZE:
                        continue

                    # ECG Info 파싱
                    ecg_info_bytes = values[ECG_INFO_INDEX:ECG_INFO_INDEX + 3]
                    p_info = ecg_info_bytes[0]
                    p1 = ecg_info_bytes[1]
                    p2 = ecg_info_bytes[2]

                    p1_peak_info = p_info & 0b111  # 정수 값 (0~7)
                    peak_index = p1 & 0b111111     # 0~63
                    asys_flag = (p1 >> 6) & 0b1    # Asystole 여부

                    # 조건 완화
                    label_class = p1_peak_info if (not asys_flag and 0 <= peak_index < 50) else None
                    label_index = peak_index if label_class is not None else None

                    # ECG 파형 변환
                    ecg_waveform = convert_ecg_waveform(values[ECG_WAVEFORM_INDEX : ECG_WAVEFORM_INDEX + 100])
                    packet_buffer.append((ecg_waveform, label_index, label_class))

                    if len(packet_buffer) < WINDOW_SIZE:
                        continue

                    # 250 샘플짜리 window 생성
                    window_waveform = []
                    window_label = [0] * (50 * WINDOW_SIZE)

                    for i, (wave, lbl_idx, lbl_cls) in enumerate(packet_buffer[-WINDOW_SIZE:]):
                        window_waveform.extend(wave)
                        if lbl_idx is not None and lbl_cls is not None:
                            absolute_idx = i * 50 + lbl_idx
                            if 0 <= absolute_idx < len(window_label):
                                window_label[absolute_idx] = lbl_cls

                    # TFRecord 직렬화 및 저장
                    serialized = serialize_example(window_waveform, window_label, subject_id, file)
                    writer.write(serialized)

                    # 슬라이딩
                    if STRIDE == 1:
                        packet_buffer.pop(0)
                    else:
                        packet_buffer = packet_buffer[STRIDE:]

        except Exception as e:
            print(f"[오류] {file_path} → {e}")


writer.close()
print(" TFRecord 저장 완료")
