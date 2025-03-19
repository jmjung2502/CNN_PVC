from collections import defaultdict
import os

# -------- 설정 --------
BASE_PATH = r"C:\CCO_Clinic_Data\04. 고신대 성인홀터임상(임성일교수님)\01. HiCardi+ data\03. parsing data(IRB승인 대상자 100명, 2024-08-08)"
PACKET_SIZE = 156
ECG_INFO_INDEX = 11  # ECG Info 위치
# ---------------------

# Peak Info 정의
PEAK_INFO_MAP = {
    0: "NaN",
    1: "R peak (〮)",
    2: "VPC (V)",
    3: "APC (A)",
    4: "Block beat (B)",
    5: "Paced beat (P)",
    6: "Pause",
    7: "Unknown (Q)"
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
        "P1 Peak Info": PEAK_INFO_MAP.get(p1_peak_info, "Unknown"),
        "P2 Peak Info": PEAK_INFO_MAP.get(p2_peak_info, "Unknown"),
        "P1": {
            "Asystole": bool(asys_flag),
            "Peak Index": peak_index1
        },
        "P2": {
            "R position shift": bool(rpo_flag),
            "Noise": bool(noise_flag),
            "Peak Index": peak_index2
        }
    }

results = []

for folder in sorted(os.listdir(BASE_PATH)):
    folder_path = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith("_raw.txt"):
            continue

        file_path = os.path.join(folder_path, file)
        peak_counts = defaultdict(int)
        total_packets = 0

        try:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    values = list(map(int, line.strip().split()))
                    if len(values) != PACKET_SIZE:
                        continue

                    total_packets += 1
                    ecg_info_bytes = values[ECG_INFO_INDEX:ECG_INFO_INDEX + 3]
                    info = parse_ecg_info(ecg_info_bytes)

                    # P1, P2 모두 카운팅
                    peak_counts[info["P1 Peak Info"]] += 1
                    peak_counts[info["P2 Peak Info"]] += 1

            results.append({
                "file": file_path,
                "packets": total_packets,
                "peak_counts": dict(peak_counts)
            })

        except Exception as e:
            print(f"[오류] 파일 처리 실패: {file_path} -> {e}")

# 결과 출력
for r in results:
    print(f"\n {os.path.basename(r['file'])}")
    print(f"총 패킷 수: {r['packets']}")
    print("Peak 타입별 개수:")
    for peak_type, count in sorted(r['peak_counts'].items()):
        print(f"  - {peak_type}: {count}")
