# configs_shanghai.py
# Shanghai_T2DM 데이터셋용 실험 설정 파일
#
# 왜 CGMacros/HUPA용 config와 분리했는가: configs_cgmacros.py 상단 설명 참고.
# Shanghai는 특히 (1) 15분 간격, (2) 세션(방문) 단위 파일, (3) 단일 피처(CGM만)라는
# 점에서 CGMacros와도 구조가 달라 별도 config가 필요하다.

# =========================
# 데이터 경로
# =========================
# 폴더 자체가 Shanghai_T2DM으로 분리되어 있어 T1DM 데이터는 아예 참조하지 않는다
# (실험 조건: T2DM만 사용).
DATA_DIR = "./data/ShangHai Datatset/Shanghai_T2DM"

# HUPA/CGMacros의 "환자"에 대응하는 단위는 "세션 파일"(환자 1명이 방문마다 파일 하나).
# None이면 전체 세션 사용, 정수 n이면 앞에서부터 n개 세션만 사용(맥북 스모크 테스트용).
SESSION_LIMIT = None

# =========================
# 사용할 피처 정의
# =========================
# 실험 조건: 연속혈당(CGM)만 사용. 인슐린/식이 관련 컬럼은 로더에서 아예 읽지 않는다
# (원 논문이 T1D 대상이라 인슐린 투여가 핵심 피처였던 것과 달리, 이번 실험은
#  T2DM 데이터의 CGM 단일 피처만으로 재현 가능한지 보는 것이 목적).
FEATURES = ["glucose"]
COLUMN_MAP = {"glucose": 0}

# =========================
# 시간 해상도 / 윈도우 설정
# =========================
# Shanghai CGM은 15분 간격 (CGMacros의 1분, HUPA의 5분과 다름 -> 반드시 데이터셋별로 따로 관리)
SAMPLE_INTERVAL_MIN = 15

# 실험 조건 지정값: 12스텝 x 15분 = 180분(3시간) 입력창.
# 참고로 이 값은 원 논문의 3시간 입력창(HUPA 기준 36스텝x5분)과 "실제 시간 길이"는
# 동일하다 (CGMacros처럼 논문 대비 짧아지는 문제가 없음).
INPUT_TIMESTEPS = 12

# CGMacros와 동일하게 "분" 단위로 지정하고 샘플링 간격으로 스텝 수를 자동 환산한다.
HORIZON_MINUTES = 30  # 60으로 바꾸면 60분 뒤 예측 (15분 간격이므로 4스텝)
HORIZON_LENGTH = HORIZON_MINUTES // SAMPLE_INTERVAL_MIN

# 윈도우(L)+예측시점(H)조차 못 채우는 너무 짧은 세션은 자동 제외.
# +10은 train/val/test로 쪼갰을 때 각 split이 최소한 윈도우 하나는 만들 수 있도록 하는 여유분.
MIN_SESSION_ROWS = INPUT_TIMESTEPS + HORIZON_LENGTH + 10

# =========================
# 학습 설정 (실험 조건 지정값)
# =========================
BATCH_SIZE = 1024
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
LAMBDA_REG = 0.01
REG_TERM = "kl"

# 본 학습은 서버(CUDA)에서 돌리기로 해서 기본값을 cuda로 둔다.
# utils.get_device(preferred)가 지정한 backend를 못 쓰면 mps -> cuda -> cpu 순으로
# 자동 폴백하므로, 맥북에서 스모크 테스트할 땐 이 값을 "mps"로만 바꾸면 됨.
DEVICE = "cuda"  # "mps" | "cuda" | "cpu"

# =========================
# 모델 설정 (TEM, model.py의 e_Transformers 그대로 재사용)
# =========================
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
MAX_LEN = 100

# =========================
# 결과 저장 경로
# =========================
MODEL_SAVE_PATH = "./shanghai_tem_model.pth"
LOSS_SAVE_PATH = "./shanghai_training_loss_plot.png"
ECP_GRAPH_SAVE_PATH = "./shanghai_ecp_graph_plot.png"
