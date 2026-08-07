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
HORIZON_MINUTES = 60  # AUTO-SET by run_remaining_experiments.sh
HORIZON_LENGTH = HORIZON_MINUTES // SAMPLE_INTERVAL_MIN

# 윈도우(L)+예측시점(H)조차 못 채우는 너무 짧은 세션은 자동 제외.
# +10은 train/val/test로 쪼갰을 때 각 split이 최소한 윈도우 하나는 만들 수 있도록 하는 여유분.
MIN_SESSION_ROWS = INPUT_TIMESTEPS + HORIZON_LENGTH + 10

# =========================
# 이동평균(MA) 정책
# =========================
# CGMacros와 동일한 원칙(cgmacros/configs_cgmacros.py 참고): 입력(X)에만 과거방향(causal)
# 이동평균을 적용하고, 타깃(Y)은 항상 원본을 유지한다 (APPLY_MA_TO_Y=False 고정).
#
# 8/4 이전까지는 Shanghai를 MA 미적용 상태로 300epoch 학습했었다(그 결과는
# ./shanghai_h30/h60_tem_model.pth로 이미 저장/커밋되어 있음 -> 이번 스윕에서도 그대로 재사용).
#
# 8/7 결정: 기존 고정값(67행=1005분)이 임의로 너무 길었다는 판단 하에 재설계.
# CGMacros와 동일하게 "분" 단위 후보 30/60/180으로 통일하고, apply_causal_moving_average()가
# 받는 window는 "행 개수"이므로 여기서 분 -> 행으로 환산한다(SAMPLE_INTERVAL_MIN=15이므로
# 30/60/180분은 각각 2/4/12행). 후보마다 개별 실행 대신, train_shanghai.py 한 번의 실행
# 안에서 이 리스트를 순회하며 각각 300epoch씩 정식 학습한다.
MA_WINDOW_CANDIDATES_MINUTES = [30, 60, 180]
APPLY_MA_TO_Y = False  # 타깃은 항상 원본 유지 (임의로 True로 바꾸지 말 것)


def ma_minutes_to_rows(ma_window_minutes):
    """분 단위 MA 후보를 apply_causal_moving_average()가 받는 '행 개수'로 환산.
    Shanghai는 15분 그리드라 CGMacros(1분 그리드)와 달리 분==행이 아니므로 변환이 필요하다."""
    if not ma_window_minutes:
        return None
    return ma_window_minutes // SAMPLE_INTERVAL_MIN

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
# 8/7부터는 MA 적용 여부가 True/False 하나가 아니라 "몇 분짜리 MA인지"까지 파일명에
# 들어가야 하므로(30/60/180분을 한 스크립트 실행 안에서 순회), CGMacros와 동일하게
# 고정 상수 대신 ma_window_minutes를 인자로 받는 함수로 바꿨다. None/0을 넘기면 MA 미적용
# 경로가 나온다 (기존 shanghai_h30/h60_tem_model.pth와 그대로 호환).
_H_SUFFIX = f"_h{HORIZON_MINUTES}"


def _ma_suffix(ma_window_minutes):
    """MA 윈도우 값(분)을 파일명 접미사로 변환. None/0이면 '이동평균 미적용'을 뜻하는 빈 문자열."""
    return f"_ma{ma_window_minutes}" if ma_window_minutes else ""


def model_save_path(ma_window_minutes=None):
    return f"./shanghai{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_tem_model.pth"


def loss_save_path(ma_window_minutes=None):
    return f"./shanghai{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_training_loss_plot.png"


def ecp_graph_save_path(ma_window_minutes=None):
    return f"./shanghai{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_ecp_graph_plot.png"
