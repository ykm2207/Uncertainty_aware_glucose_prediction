import os
import pandas as pd
import matplotlib.pyplot as plt

# cgmacros/ 폴더로 이동하면서, 결과 이미지(ma_check.png)가 어디서 실행하든
# 항상 이 스크립트와 같은 폴더에 저장되도록 저장 경로를 스크립트 기준 상대경로로 고정.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) 파일 경로 — 실제 경로로 수정
path = "/Users/yoo/Uncertainty_aware_glucose_prediction/data/CGMacros/CGMacros-001/CGMacros-001.csv"
df = pd.read_csv(path)

# 2) 컬럼명 먼저 확인 (한 번만 보고 지워도 됨)
print(df.columns.tolist())

# 3) 혈당 컬럼 지정 — 위 출력 보고 실제 이름으로 수정
col = "Libre GL"

raw = df[col].values[:3000]          # 앞 3000분(약 50시간)만
ma200 = pd.Series(raw).rolling(200, center=True).mean()

plt.figure(figsize=(14, 5))
plt.plot(raw, label="원본 (1분 보간)", alpha=0.6)
plt.plot(ma200, label="200점 이동평균", linewidth=2)
plt.xlabel("시간 (분)")
plt.ylabel("혈당 (mg/dL)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ma_check.png"), dpi=150)   # 발표 자료용으로 저장
plt.show()