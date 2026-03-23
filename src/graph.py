import re
import pandas as pd
import matplotlib.pyplot as plt
import os

log_folder = '/home/autonav/lidar_paper_ws'

def parse_log_file(file_name):
    try:
        with open(file_name, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"{file_name} 파일이 없어요!")
        return None
    
    # 데이터 추출 (Regex)
    timestamps = re.findall(r'\[\s*(\d+\.\d+)\]: 0\)', content)
    roi_points = re.findall(r'1\) LiDAR ROI Points\s*:? (\d+)', content)
    efficiencies = re.findall(r'2\) Data Efficiency\s+: ([\d\.]+)%', content)
    
    min_len = min(len(timestamps), len(roi_points), len(efficiencies))
    
    df = pd.DataFrame({
        'Time': pd.to_numeric(timestamps[:min_len]),
        'Points': pd.to_numeric(roi_points[:min_len]),
        'Efficiency': pd.to_numeric(efficiencies[:min_len])
    })
    
    if not df.empty:
        df['Time'] = df['Time'] - df['Time'].iloc[0]
    return df

# 1. 비교할 파일 리스트와 라벨 설정 
files = {
    os.path.join(log_folder, 'dynamic_final_count_2.txt'): 'Semantic ROI',
    os.path.join(log_folder, 'fixed_final_count_2.txt'): 'Geometric ROI',
    os.path.join(log_folder, 'waypoint_final_count_2.txt'): 'Waypoint ROI'  
}

# --- 그래프 1: Efficiency 비교 (따로 저장) ---
plt.figure(figsize=(10, 6))
for file_name, label in files.items():
    data = parse_log_file(file_name)
    if data is not None:
        plt.plot(data['Time'], data['Efficiency'], label=label, linewidth=2)

plt.axvspan(22.9, 29, color='gray', alpha=0.15, label= 'Intersection')
plt.axvspan(44.7, 51, color='gray', alpha=0.15)
plt.axvspan(65, 72, color='gray', alpha=0.15)
plt.axvspan(86.5, 93, color='gray', alpha=0.15)

plt.title('Data Retention Rate Comparison', fontsize=15)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Data Retention Rate (%)', fontsize=12)
#plt.legend()
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Efficiency_Comparison_final_2_3.png', dpi=300) # 고화질(300dpi) 저장
plt.close() # 메모리 확보를 위해 닫기
print("Efficiency 그래프 저장 완료: Efficiency_Comparison.png")

'''
# --- 그래프 2: ROI Points 비교 (따로 저장) ---
plt.figure(figsize=(10, 6))
for file_name, label in files.items():
    data = parse_log_file(file_name)
    if data is not None:
        plt.plot(data['Time'], data['Points'], label=label, linewidth=2)

plt.title('LiDAR ROI Points Comparison', fontsize=15)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Point Count', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('ROI_Points_Comparison.png', dpi=300) # 고화질(300dpi) 저장
plt.close()
print("✅ ROI Points 그래프 저장 완료: ROI_Points_Comparison.png")
'''