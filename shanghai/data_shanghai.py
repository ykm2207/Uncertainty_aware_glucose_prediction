# data_shanghai.py
# Shanghai_T2DM 데이터셋 전용 전처리 파이프라인
#
# 왜 별도 파일인가: data.py(HUPA, ";" 구분 CSV)나 data_cgmacros.py(1분 간격, 결측
# 세그먼트 처리)와 달리 Shanghai는 세션(방문)별 Excel 파일이고, 파일 포맷이
# 두 가지(.xls/.xlsx)로 섞여 있고, 컬럼명도 파일마다 조금씩 다르다(예: 식이 컬럼이
# '饮食' 또는 '进食量'로 다름). 이런 원본 특이사항을 흡수하는 로더가 필요해서
# 분리했다. 반면 make_windows/split_data/extract_features 같은 순수 계산 함수는
# data.py에서 그대로 재사용한다(중복 구현 방지).
import os
import sys
import glob
import numpy as np
import pandas as pd

# shanghai/ 폴더에서도 루트의 공용 data.py(make_windows 등)를 그대로 쓰기 위해
# 루트 디렉터리를 sys.path에 추가 (repo 루트에서 실행하는 걸 기준으로 함).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import make_windows, split_data, extract_features
from utils import apply_causal_moving_average


def discover_sessions(data_dir, session_limit=None):
    """
    data_dir 아래의 세션 파일(.xlsx/.xls)을 찾는다.

    왜 이렇게 필터링하는가:
        압축 해제 과정에서 생긴 'Zone.Identifier'(윈도우 대체 데이터 스트림 흔적),
        '~$'로 시작하는 Excel 임시 잠금 파일이 실제 데이터 파일과 함께 섞여 있어서
        그대로 glob하면 읽을 수 없는 파일까지 걸린다. 이 둘을 명시적으로 제외한다.

    Args:
        data_dir: Shanghai_T2DM 폴더 경로
        session_limit: None이면 전체, 정수 n이면 앞에서부터 n개 세션만 사용(스모크 테스트용)

    Returns:
        [(session_id: str, path: str), ...] (session_id = 파일명, 예: "2001_0_20201102")
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.xls")))
    paths = [p for p in paths if "Zone.Identifier" not in p and not os.path.basename(p).startswith("~$")]
    sessions = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    if session_limit is not None:
        sessions = sessions[:session_limit]
    return sessions


def load_single_session_shanghai(path):
    """
    Shanghai 세션 Excel 1개를 읽어 시간순 정렬된 CGM 값만 뽑는다.

    왜 컬럼명을 부분일치로 찾는가:
        원본 파일이 최소 3가지 컬럼셋 변형을 가진다(식이 컬럼명이 '饮食'/'진식량' 등으로
        다름). 우리가 실제로 쓰는 건 'CGM (mg / dl)'과 'Date' 뿐이라, 정확한 전체
        문자열이 아니라 'CGM'이 포함된 컬럼을 찾는 방식으로 파일 간 표기 차이를 흡수한다.
        인슐린/식이 컬럼은 아예 읽지 않으므로(실험 조건: CGM만 사용) 그 변형은
        영향을 주지 않는다.

    Returns:
        pandas.DataFrame, 컬럼=["glucose"], Date 기준 오름차순 정렬, 결측 제거 완료.
    """
    df = pd.read_excel(path)

    date_col = next((c for c in df.columns if "Date" in c), None)
    cgm_col = next((c for c in df.columns if "CGM" in c), None)
    if date_col is None or cgm_col is None:
        raise ValueError(f"{path}: 'Date' 또는 'CGM' 컬럼을 찾지 못함 (컬럼: {df.columns.tolist()})")

    out = df[[date_col, cgm_col]].rename(columns={date_col: "Date", cgm_col: "glucose"})
    out = out.dropna(subset=["glucose"]).sort_values("Date").reset_index(drop=True)
    return out[["glucose"]]


def report_session_lengths(data_dir, min_session_rows):
    """
    세션별 유효 CGM 행 수 분포를 출력하는 진단 스크립트.

    왜 필요한가: 세션 길이가 방문마다 제각각이라(수백~수천 행), L=12+H 만큼도
    못 채우는 너무 짧은 세션이 섞여 있을 수 있다. 이 함수로 실제 학습 전에
    몇 개 세션이 제외되는지 미리 확인한다.
    """
    sessions = discover_sessions(data_dir)
    rows = []
    for sid, path in sessions:
        try:
            df = load_single_session_shanghai(path)
        except Exception as e:
            print(f"  [읽기 실패] {sid}: {e}")
            continue
        rows.append({
            "세션": sid,
            "유효행수": len(df),
            "실제시간(시간)": round(len(df) * 15 / 60, 1),
            "사용가능": len(df) >= min_session_rows,
        })

    report_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"=== Shanghai_T2DM 세션 길이 진단 (최소 {min_session_rows}행 필요) ===")
    print(f"전체 세션 {len(sessions)}개 중 읽기 성공 {len(report_df)}개")
    print(f"사용 가능(길이 충분) 세션 수: {report_df['사용가능'].sum()}개")
    too_short = report_df.loc[~report_df["사용가능"], "세션"].tolist()
    print(f"너무 짧아 제외되는 세션: {too_short}")
    return report_df


def _concat_or_empty(arrays, fallback_shape):
    if len(arrays) == 0:
        return np.zeros(fallback_shape)
    return np.concatenate(arrays, axis=0)


def prepare_dataset_shanghai(data_dir, feature_names, column_map, L, H,
                              session_limit=None, min_session_rows=None,
                              ma_window=None, apply_ma_to_y=False):
    """
    전체 Shanghai 세션을 로드 -> (환자가 아니라) 세션 단위로 train/val/test 분할 ->
    (필요시 이동평균) -> 윈도우 생성 -> 전체 세션에 대해 concat한다.

    왜 "환자"가 아니라 "세션" 단위로 도는가:
        같은 환자가 여러 번 방문했어도 방문 사이에는 몇 주~몇 달의 공백이 있어
        하나의 연속된 시계열로 볼 수 없다. 그래서 data.py의 prepare_dataset()이
        "환자 1명 = CSV 1개 = 연속 시계열"로 가정하고 도는 것과 동일한 패턴을
        "세션 1개 = 연속 시계열" 단위로 그대로 적용한다.

    이동평균은 cgmacros/data_cgmacros.py의 prepare_dataset_cgmacros()와 완전히 동일한
    원칙을 따른다: raw(원본)와 smoothed(스무딩) 두 배열을 각각 만들어서, X는 스무딩된
    쪽에서, Y(타깃)는 apply_ma_to_y=False(기본값, 권장)면 raw 쪽에서 추출한다.
    Shanghai는 피처가 glucose 하나뿐이라 "X도 결국 glucose, Y도 glucose"이지만,
    이 둘을 서로 다른 배열(스무딩 여부만 다름)에서 뽑는다는 점이 핵심이다.
    """
    splits = {"train": ([], []), "val": ([], []), "test": ([], []), "train_val": ([], [])}

    sessions = discover_sessions(data_dir, session_limit)
    n_skipped_short = 0
    n_skipped_error = 0

    for sid, path in sessions:
        try:
            df = load_single_session_shanghai(path)
        except Exception as e:
            print(f"  [건너뜀] {sid}: 읽기 실패 ({e})")
            n_skipped_error += 1
            continue

        raw_segment = df.to_numpy(dtype=float)
        if min_session_rows is not None and len(raw_segment) < min_session_rows:
            n_skipped_short += 1
            continue

        smoothed_segment = (
            apply_causal_moving_average(raw_segment, ma_window)
            if ma_window is not None else raw_segment
        )

        raw_train, raw_val, raw_test, raw_train_val = split_data(raw_segment)
        sm_train, sm_val, sm_test, sm_train_val = split_data(smoothed_segment)

        for key, raw_split, sm_split in (
            ("train", raw_train, sm_train), ("val", raw_val, sm_val),
            ("test", raw_test, sm_test), ("train_val", raw_train_val, sm_train_val),
        ):
            if len(raw_split) < L + H:
                continue
            X, _ = extract_features(sm_split, feature_names, column_map)
            Y_source = sm_split if apply_ma_to_y else raw_split
            _, Y = extract_features(Y_source, feature_names, column_map)
            Xw, Yw = make_windows(X, Y, L, H)
            if len(Xw) == 0:
                continue
            splits[key][0].append(Xw)
            splits[key][1].append(Yw)

    print(f"  세션 {len(sessions)}개 중 짧아서 제외 {n_skipped_short}개, 읽기 실패 {n_skipped_error}개")

    n_features = len(feature_names)
    result = {}
    for key, (xs, ys) in splits.items():
        X_cat = _concat_or_empty(xs, (0, L, n_features))
        Y_cat = _concat_or_empty(ys, (0, H))
        result[key] = (X_cat, Y_cat)
        print(f"  [{key}] 윈도우 {X_cat.shape[0]}개 생성")

    return result


def compute_means_variances_shanghai(data_dir, feature_names, column_map, L, H,
                                      session_limit=None, min_session_rows=None,
                                      ma_window=None):
    """
    정규화용 평균/표준편차를 train 구간에서만 계산한다 (data_cgmacros의 동명 함수와 동일한 목적/원칙).
    ma_window는 prepare_dataset_shanghai()에 준 값과 반드시 같아야 한다 - 모델이 실제로
    보는 입력(X, 항상 스무딩된 쪽)을 기준으로 통계를 내야 하기 때문.
    """
    sessions = discover_sessions(data_dir, session_limit)
    train_X_list = []

    for sid, path in sessions:
        try:
            df = load_single_session_shanghai(path)
        except Exception:
            continue
        raw_segment = df.to_numpy(dtype=float)
        if min_session_rows is not None and len(raw_segment) < min_session_rows:
            continue

        smoothed_segment = (
            apply_causal_moving_average(raw_segment, ma_window)
            if ma_window is not None else raw_segment
        )

        train_data, _, _, _ = split_data(smoothed_segment)
        if len(train_data) < L + H:
            continue
        X_train, _ = extract_features(train_data, feature_names, column_map)
        train_X_list.append(X_train)

    if not train_X_list:
        raise ValueError("정규화 통계를 계산할 train 데이터가 없습니다 (세션 필터를 확인하세요).")

    trainingset = np.concatenate(train_X_list, axis=0)
    mu_gen = np.mean(trainingset, axis=0)
    sigma_gen = np.std(trainingset, axis=0)
    mu_g = mu_gen[0]     # feature_names[0] == "glucose" 관례를 그대로 따름
    sigma_g = sigma_gen[0]

    return mu_g, sigma_g, mu_gen, sigma_gen
