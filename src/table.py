import re
import pandas as pd
import os

# 1. 파일들이 들어있는 폴더 경로
log_folder = '/home/autonav/lidar_paper_ws'

def parse_log_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    
    # 정규표현식으로 데이터 추출 (타임스탬프, 점 개수, 효율성)
    roi_points = re.findall(r'1\) LiDAR ROI Points\s*:? (\d+)', content)
    efficiencies = re.findall(r'2\) Data Efficiency\s+: ([\d\.]+)%', content)
    
    min_len = min(len(roi_points), len(efficiencies))
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Points': pd.to_numeric(roi_points[:min_len]),
        'Efficiency': pd.to_numeric(efficiencies[:min_len])
    })
    return df

# 2. 비교할 파일 명단 (왕자님의 실제 파일명과 라벨)
files = {
    os.path.join(log_folder, 'dynamic_final_count_2.txt'): 'Dynamic ROI',
    os.path.join(log_folder, 'fixed_final_count_2.txt'): 'Fixed ROI',
    os.path.join(log_folder, 'waypoint_final_count_2.txt'): 'Waypoint ROI'
}

# 3. 통계 데이터를 모을 리스트
table_rows = []

for file_path, label in files.items():
    data = parse_log_file(file_path)
    if data is not None and not data.empty:
        # 각 항목별 통계 계산
        table_rows.append({
            'Method': label,
            'Avg Points': round(data['Points'].mean(), 1),
            'Max Points': data['Points'].max(),
            'Min Points': data['Points'].min(),
            'Avg Efficiency (%)': f"{data['Efficiency'].mean():.2f}%"
        })

# 4. 표 생성 및 출력
summary_table = pd.DataFrame(table_rows)

print("\n" + "="*65)
print("             [ LiDAR ROI Points Statistics Table ]")
print("="*65)
if not summary_table.empty:
    print(summary_table.to_string(index=False))
else:
    print("데이터를 찾을 수 없습니다. 파일 경로와 형식을 확인해 주세요!")
print("="*65)

# 5. CSV 파일로 저장 (엑셀용)
summary_table.to_csv('ROI_Table_final_Result_2.csv', index=False)
print("\n✅ 요약 표가 'ROI_Table_final_Result_2.csv'로 저장되었습니다!")