# configs_cgmacros.py
# CGMacros 데이터셋용 실험 설정 파일
#
# 왜 configs.py(HUPA용)와 분리했는가:
#   HUPA-UCM(5분 간격), CGMacros(1분 간격), Shanghai(15분 간격)는 샘플링 주기와
#   보유 피처가 전혀 다르다. 하나의 config로 억지로 묶으면 조건 분기가 늘어나서
#   실수하기 쉬우므로, 데이터셋별로 완전히 분리된 설정 파일을 둔다.
#   (data.py / model.py / utils.py의 순수 유틸/모델 함수는 그대로 재사용한다.)

# =========================
# 데이터 경로 및 대상 환자
# =========================
DATA_DIR = "./data/CGMacros"

# prepare_data_revision.py로 만든, 긴 결측 구간을 미리 잘라낸 복사본 경로.
# train_cgmacros_revision.py / evaluate_cgmacros_revision.py가 이 경로를 사용한다.
# (원본 DATA_DIR 파이프라인은 그대로 두고 비교할 수 있도록 별도 상수로 분리)
DATA_DIR_REVISION = "./data_revision/CGMacros"

# HUPA처럼 환자 ID를 하드코딩하지 않고 data_cgmacros.discover_patients()가
# 폴더를 스캔해서 자동으로 찾는다. PATIENT_LIMIT을 정수로 주면 앞에서부터
# 그 수만큼만 사용 -> 맥북에서 빠르게 파이프라인 동작만 확인할 때 사용.
# (예: PATIENT_LIMIT = 3 으로 두면 3명 데이터로만 스모크 테스트)
PATIENT_LIMIT = None

# =========================
# 사용할 피처 정의
# =========================
# 실험 조건: Dexcom GL은 미사용(Libre GL만 사용), 인슐린 관련 컬럼은
# CGMacros 원본에 아예 없으므로 자동으로 제외됨.
# COLUMN_MAP은 data_cgmacros.load_single_patient_cgmacros()가 만드는
# numpy 배열의 컬럼 순서(0=glucose, 1=heart_rate, 2=calories)와 반드시 일치해야 한다.
FEATURES = ["glucose", "heart_rate", "calories"]

COLUMN_MAP = {
    "glucose": 0,     # Libre GL
    "heart_rate": 1,  # HR
    "calories": 2,    # Calories (Activity) -- CGMacros-007은 원본에 이 컬럼이 없어 NaN으로 채워짐
}

# 원본 CGMacros CSV 컬럼명 -> 우리가 쓰는 내부 피처 이름 매핑
# (raw CSV 컬럼명이 그대로 FEATURES 순서가 아니므로 로더에서 이 매핑을 사용)
RAW_COLUMN_NAMES = {
    "glucose": "Libre GL",
    "heart_rate": "HR",
    "calories": "Calories (Activity)",
}

# =========================
# 시간 해상도 / 윈도우 설정
# =========================
# CGMacros 원본 CSV는 1분 간격 그리드다 (Libre GL 자체는 15분 실측을 1분 단위로
# 선형보간한 값 - 진짜 측정 주기는 15분). 원 논문(HUPA)은 5분 간격을 썼으므로,
# NATIVE(원본 그리드)와 SAMPLE(리샘플링 후 실제로 학습에 쓰는 간격)을 구분해서 관리한다.
NATIVE_SAMPLE_INTERVAL_MIN = 1

# 원 논문과 샘플링 간격을 맞추기 위해 5분으로 다운샘플링한다 (data_cgmacros.decimate_segment
# 참고). RESAMPLE_FACTOR = SAMPLE_INTERVAL_MIN // NATIVE_SAMPLE_INTERVAL_MIN 만큼
# 1분 그리드에서 몇 번째 행마다 하나씩 뽑을지를 뜻한다 (5 -> 5행마다 1개).
# ⚠️ 리샘플링해도 5분 그리드 값의 2/3는 여전히 보간값이다(진짜 측정 주기가 15분이라서).
# 이 한계는 리샘플링으로 해결되지 않으므로 결과 보고 시 반드시 같이 명시할 것.
SAMPLE_INTERVAL_MIN = 5
RESAMPLE_FACTOR = SAMPLE_INTERVAL_MIN // NATIVE_SAMPLE_INTERVAL_MIN

# 실험 조건에서 지정된 입력 타임스텝 수. 이제 SAMPLE_INTERVAL_MIN=5이므로
# 36스텝 x 5분 = 180분(3시간) 입력창 -> 원 논문(5분 x 36스텝 = 3시간)과 시간 범위가
# 정확히 일치한다 (예전엔 1분 그리드 그대로 써서 36분=논문의 1/5 수준이었음).
INPUT_TIMESTEPS = 36

# 예측 시점(horizon)을 "분" 단위로 지정하고, 실제 스텝 수는 샘플링 간격으로
# 자동 환산한다. 이렇게 하면 데이터셋마다 샘플링 간격이 달라도 "30분 뒤 예측"이라는
# 의미가 코드 전역에서 일관되게 유지된다 (하드코딩된 스텝 수를 쓰면 헷갈리기 쉬움).
HORIZON_MINUTES = 60  # AUTO-SET by run_ma_sweep.sh
HORIZON_LENGTH = HORIZON_MINUTES // SAMPLE_INTERVAL_MIN

# =========================
# 결측치(HR/Calories) 처리 정책
# =========================
# 실험 조건 메모: "HR/칼로리 결측구간을 절단하고도 쓸 수 있는지" 확인이 필요한 상태.
# 이 값 이하로 짧게 끊긴 결측은 선형보간으로 메우고, 이 값을 넘는(=오래 기기를
# 빼놓은 것으로 추정되는) 결측 구간은 그 지점에서 시계열을 아예 끊어
# 별도의 연속 구간(segment)으로 분리한다. 서로 무관한 시간대를 이어붙여
# 하나의 윈도우로 만드는 것을 막기 위함. data_cgmacros.report_missingness()로
# 환자별 결측 패턴을 먼저 확인하고 이 값을 조정할 것.
# 이 절단 판정은 항상 NATIVE_SAMPLE_INTERVAL_MIN(1분) 그리드 기준으로 먼저 이뤄지고,
# 그 이후에 RESAMPLE_FACTOR만큼 다운샘플링한다 (순서가 바뀌면 분 단위 임계값 계산이 틀어짐).
MAX_GAP_FOR_INTERP_MIN = 15

# =========================
# 이동평균(MA) 정책
# =========================
# 8/2 실측 검증: 이동평균을 타깃(Y)에도 적용하면 다수 환자에서 저혈당 이벤트가
# 완전히 사라져서 임상 지표가 무의미해짐 -> 그래서 입력(X)에만 적용하고 타깃은
# 항상 원본 그대로 둔다(APPLY_MA_TO_Y=False 고정, data_cgmacros.apply_causal_moving_average
# 참고). 방향도 미래 정보 누출을 막기 위해 항상 과거방향(causal)만 사용한다.
#
# 8/4 결정: 팀 지정값 200으로 먼저 300epoch 학습해봤는데(결과: cgmacros_revision_ma_*),
# MA 미적용 대비 RMSE/MAE/MARD/Zone 정확도/민감도가 전부 뚜렷하게 나빠짐 (Shanghai도 동일 경향).
# MA 미적용 버전은 이미 300epoch 정식으로 완료돼 있으므로(cgmacros_revision_h30/h60) 그대로 재사용한다.
#
# 8/7 결정: 200분 고정값이 임의로 너무 길었다는 판단 하에 이동평균 실험을 다시 설계.
# 30/60/180분 세 후보로 바꿔서, 후보마다 개별로 학습을 다시 돌리는 대신 스크립트 한 번
# 실행(train_cgmacros_revision.py)에서 이 리스트를 순회하며 후보마다 300epoch씩 정식
# 학습하도록 바꿨다(아래 파일 하단 "결과 저장 경로" 섹션의 경로 생성 함수 참고).
# 단위는 "분"이며 NATIVE_SAMPLE_INTERVAL_MIN(1분) 그리드 기준으로 적용된다
# (다운샘플링 전에 적용하므로 5분 그리드 행 개수가 아니라 실제 분 단위로 정확히 적용됨).
MA_WINDOW_CANDIDATES_MINUTES = [30, 60, 180]
APPLY_MA_TO_Y = False  # 타깃은 항상 원본 유지 (위 설명 참고, 임의로 True로 바꾸지 말 것)

# =========================
# 학습 설정 (실험 조건 지정값)
# =========================
BATCH_SIZE = 1024
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
LAMBDA_REG = 0.01
REG_TERM = "kl"

# "mps" | "cuda" | "cpu" 중 우선 시도할 backend. 본 학습은 서버(CUDA)에서 돌리기로
# 해서 기본값을 cuda로 둔다. 맥북에서 파이프라인만 빠르게 스모크 테스트할 땐
# 이 값을 "mps"로 바꿔서 실행하면 됨. utils.get_device(preferred)는 지정한 backend를
# 최우선으로 쓰되, 그게 이 기기에서 안 되면 mps -> cuda -> cpu 순으로 자동 폴백한다
# (예: 이 값을 cuda로 둔 채 맥북에서 실수로 돌려도 mps로 알아서 넘어감).
DEVICE = "cuda"

# =========================
# 모델 설정 (TEM, model.py의 e_Transformers 그대로 재사용)
# =========================
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
MAX_LEN = 100

# =========================
# 결과 저장 경로 (원본 HUPA 파이프라인의 산출물과 겹치지 않도록 파일명 분리)
# =========================
# 8/7부터는 MA 적용 여부가 True/False 하나가 아니라 "몇 분짜리 MA인지"까지 파일명에
# 들어가야 하므로(30/60/180분을 한 스크립트 실행 안에서 순회), 고정 상수 대신
# ma_window_minutes를 인자로 받는 함수로 바꿨다. None/0을 넘기면 MA 미적용 경로가 나온다
# (기존 cgmacros_revision_h30_tem_model.pth 같은 MA 미적용 완료본과 그대로 호환).
_H_SUFFIX = f"_h{HORIZON_MINUTES}"


def _ma_suffix(ma_window_minutes):
    """MA 윈도우 값(분)을 파일명 접미사로 변환. None/0이면 '이동평균 미적용'을 뜻하는 빈 문자열."""
    return f"_ma{ma_window_minutes}" if ma_window_minutes else ""


def model_save_path(ma_window_minutes=None):
    return f"./cgmacros{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_tem_model.pth"


def loss_save_path(ma_window_minutes=None):
    return f"./cgmacros{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_training_loss_plot.png"


def ecp_graph_save_path(ma_window_minutes=None):
    return f"./cgmacros{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_ecp_graph_plot.png"


# data_revision(절단 복사본) 학습 결과물 경로 -- 원본(raw) 파이프라인 결과와
# 섞이지 않도록 파일명을 다르게 둔다. train_cgmacros_revision.py/evaluate_cgmacros_revision.py 전용.
def model_save_path_revision(ma_window_minutes=None):
    return f"./cgmacros_revision{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_tem_model.pth"


def loss_save_path_revision(ma_window_minutes=None):
    return f"./cgmacros_revision{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_training_loss_plot.png"


def ecp_graph_save_path_revision(ma_window_minutes=None):
    return f"./cgmacros_revision{_ma_suffix(ma_window_minutes)}{_H_SUFFIX}_ecp_graph_plot.png"
