# data_cgmacros.py
# CGMacros 데이터셋 전용 전처리 파이프라인
#
# 왜 data.py를 그대로 안 쓰고 새로 만들었나:
#   data.py의 load_single_patient/prepare_dataset은 HUPA-UCM CSV 포맷(";" 구분자,
#   고정된 컬럼 7개, 결측치 없음)에 맞춰져 있다. CGMacros는 구분자도 다르고,
#   피처 구성도 다르고(Dexcom GL 미사용, 인슐린 없음), 무엇보다 HR/Calories에
#   결측 구간이 실제로 존재해서(최대 연속 결측 14시간대) 그냥 이어붙이면 안 된다.
#   반면 make_windows/split_data/extract_features/compute_means_variances의 "본계산" 부분은
#   데이터셋과 무관한 순수 함수이므로 data.py에서 그대로 가져다 쓴다(중복 구현 방지).
import os
import glob
import numpy as np
import pandas as pd

from data import make_windows, split_data, extract_features


def discover_patients(data_dir, patient_limit=None):
    """
    data_dir 아래의 CGMacros-XXX/CGMacros-XXX.csv 파일들을 스캔해서 환자 목록을 만든다.

    왜 필요한가:
        원본 HUPA 파이프라인(configs.py의 PATIENT_IDS)은 환자 ID를 코드에 하드코딩했다.
        CGMacros는 환자 번호가 001~049 사이에 군데군데 빠져 있고(004,024,025,037,040,
        050 등 결번 다수) 앞으로 데이터가 추가/제외될 수도 있으므로, 폴더를 직접 스캔해서
        실제로 존재하는 환자만 자동으로 찾는 방식이 더 안전하다.

    Args:
        data_dir: CGMacros 루트 폴더 경로 (예: "./data/CGMacros")
        patient_limit: None이면 전체 사용. 정수 n을 주면 앞에서부터 n명만 사용
                       (맥북에서 파이프라인만 빠르게 검증할 때 씀 -> configs_cgmacros.PATIENT_LIMIT)

    Returns:
        [(patient_id: str, csv_path: str), ...] 리스트 (patient_id 오름차순 정렬)
    """
    pattern = os.path.join(data_dir, "CGMacros-*", "CGMacros-*.csv")
    paths = sorted(glob.glob(pattern))
    patients = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    if patient_limit is not None:
        patients = patients[:patient_limit]
    return patients


def load_single_patient_cgmacros(csv_path, feature_names, raw_column_names):
    """
    CGMacros 환자 1명의 CSV를 읽어 우리가 쓸 피처만 뽑아 DataFrame으로 반환한다.

    왜 이렇게 구현했나:
        - CGMacros CSV는 환자마다 컬럼 구성이 조금씩 다르다(예: CGMacros-007은
          'Calories (Activity)' 컬럼이 아예 없고 'Steps'만 있음). 컬럼이 없으면
          KeyError로 죽는 대신 전부 NaN인 컬럼으로 채우고 경고를 출력한다.
          -> 이후 segment_by_gaps()에서 이 NaN이 자동으로 "사용 불가 구간"으로
             처리되어, 해당 피처가 없는 환자는 자연스럽게 학습에서 제외된다.
        - feature_names 순서를 강제로 맞춰서 반환한다. configs_cgmacros.COLUMN_MAP이
          "glucose":0, "heart_rate":1, "calories":2 처럼 위치 인덱스로 매핑되어 있으므로
          여기서 순서가 어긋나면 이후 extract_features()가 엉뚱한 컬럼을 target/입력으로
          잘못 뽑게 된다.

    Args:
        csv_path: 환자 CSV 경로
        feature_names: 내부 피처 이름 리스트, 예) ["glucose","heart_rate","calories"]
        raw_column_names: {내부 피처 이름: 원본 CSV 컬럼명} 매핑 (configs_cgmacros.RAW_COLUMN_NAMES)

    Returns:
        pandas.DataFrame, 컬럼 = feature_names 순서, 인덱스 = 시간순 정렬된 0..N-1
        (1분 간격 그리드이므로 행 번호 간격 = 1분과 동일하다고 취급)
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    cols = {}
    for feat in feature_names:
        raw_name = raw_column_names[feat]
        if raw_name in df.columns:
            cols[feat] = df[raw_name].astype(float)
        else:
            print(f"  [경고] {os.path.basename(csv_path)}: '{raw_name}' 컬럼이 없어 전부 결측으로 처리")
            cols[feat] = pd.Series(np.nan, index=df.index)

    feature_df = pd.DataFrame(cols)[feature_names]
    return feature_df


def _find_nan_runs(mask):
    """mask(bool 배열)에서 True(=결측)가 연속되는 구간을 [시작, 끝) 쌍의 리스트로 반환한다."""
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def segment_by_gaps(feature_df, max_gap_for_interp_min):
    """
    결측 구간 길이에 따라 "선형보간으로 메울지" vs "시계열을 끊을지"를 결정하고,
    최종적으로 사용 가능한 연속 구간(segment)들의 리스트를 반환한다.

    왜 이렇게 구현했나 (사용자 요청: "HR/칼로리 결측구간을 절단하고도 쓸 수 있는지" 확인):
        CGMacros는 같은 웨어러블 기기에서 HR/Calories가 몇 시간~며칠씩 통째로 빠지는
        경우가 흔하다(환자별 최대 연속 결측 중앙값 약 862분=14시간). 이런 긴 구간을
        그냥 선형보간해버리면 사실상 몇 시간짜리 가짜 데이터를 만들어내는 셈이라
        위험하다. 반대로 1~2분짜리 짧은 결측까지 전부 잘라버리면 데이터 손실이 너무 커진다.
        그래서 "max_gap_for_interp_min 이하 -> 보간, 초과 -> 절단(segment 분리)"
        기준을 두고, 그 결과를 report_missingness()로 확인해가며 임계값을 조정하도록 설계했다.
        긴 결측 이후를 짧은 결측 이전과 이어붙이지 않기 때문에, 하나의 슬라이딩 윈도우가
        서로 무관한 시간대를 섞어서 담는 일이 없다(시계열 연속성 보장).

    Args:
        feature_df: load_single_patient_cgmacros()가 반환한, 시간순 정렬된 피처 DataFrame
        max_gap_for_interp_min: 이 값(분) 이하의 연속 결측만 선형보간, 초과분은 절단

    Returns:
        [np.ndarray, ...] : 결측 없이 사용 가능한 연속 구간들의 리스트.
                             각 배열의 컬럼 순서는 feature_df와 동일.
    """
    mask = feature_df.isna().any(axis=1).values
    runs = _find_nan_runs(mask)

    # 보간 대상(짧은 결측)만 True로 표시
    short_run_mask = np.zeros(len(mask), dtype=bool)
    for start, end in runs:
        if (end - start) <= max_gap_for_interp_min:
            short_run_mask[start:end] = True

    # 전체 선형보간 값을 미리 계산해두고, "짧은 결측" 위치에만 그 보간값을 채워 넣는다.
    # (limit 옵션 없이 전체를 보간하는 이유: 짧은 구간이 배열 맨 앞/뒤에 걸쳐 있어도
    #  일관되게 처리하기 위함. 긴 결측 위치는 아래에서 다시 NaN인 채로 남겨둔다.)
    interpolated_all = feature_df.interpolate(method="linear", limit_direction="both")

    filled = feature_df.copy()
    filled.loc[short_run_mask, :] = interpolated_all.loc[short_run_mask, :]

    # 여기까지 하고도 남은 NaN = 긴 결측(절단 대상). 이 지점을 경계로 구간을 나눈다.
    still_na = filled.isna().any(axis=1).values
    valid_idx = np.where(~still_na)[0]
    if len(valid_idx) == 0:
        return []

    seg_id = np.cumsum(still_na)  # 결측 행을 지날 때마다 세그먼트 번호가 올라감
    df_valid = filled.iloc[valid_idx].copy()
    df_valid["__seg__"] = seg_id[valid_idx]

    segments = [
        g.drop(columns="__seg__").to_numpy(dtype=float)
        for _, g in df_valid.groupby("__seg__")
    ]
    return segments


def report_missingness(data_dir, feature_names, raw_column_names, max_gap_for_interp_min=15):
    """
    환자별 결측 패턴을 표로 출력하는 진단 스크립트.

    왜 필요한가:
        "HR/칼로리 결측구간을 절단해도 되는지"는 코드로 강제 결정할 문제가 아니라
        실제 데이터를 보고 판단할 문제다. 이 함수는 그 판단 근거(환자별 결측률,
        최대 연속 결측 길이, 절단 후 남는 연속 구간 수/길이)를 한눈에 보여준다.
        segment_by_gaps()의 max_gap_for_interp_min 값을 바꿔가며 이 함수를 다시
        돌려보면 "그 임계값으로 잘랐을 때 실제로 얼마나 남는지"를 바로 확인할 수 있다.
    """
    patients = discover_patients(data_dir)
    rows = []
    for pid, csv_path in patients:
        feature_df = load_single_patient_cgmacros(csv_path, feature_names, raw_column_names)
        n = len(feature_df)
        nan_ratio = feature_df.isna().any(axis=1).mean() * 100

        segments = segment_by_gaps(feature_df, max_gap_for_interp_min)
        seg_lengths = [len(s) for s in segments]

        rows.append({
            "환자": pid,
            "전체행수": n,
            "결측비율(%)": round(nan_ratio, 2),
            "절단후_구간수": len(segments),
            "절단후_최장구간(분)": max(seg_lengths) if seg_lengths else 0,
            "절단후_총사용가능행수": sum(seg_lengths),
            "사용가능비율(%)": round(sum(seg_lengths) / n * 100, 1) if n else 0.0,
        })

    report_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"=== CGMacros 결측 진단 (max_gap_for_interp_min={max_gap_for_interp_min}분 기준) ===")
    print(report_df.to_string(index=False))
    print()
    print(f"환자 평균 사용가능비율: {report_df['사용가능비율(%)'].mean():.1f}%")
    print(f"사용가능비율 0%인 환자(사실상 제외됨): "
          f"{report_df.loc[report_df['절단후_총사용가능행수'] == 0, '환자'].tolist()}")
    return report_df


def build_trimmed_copy(csv_path, out_path, max_gap_for_interp_min):
    """
    환자 CSV 원본을 읽어 HR/Calories(Activity) 결측 구간을 절단한 "복사본"을
    실제 파일로 저장한다.

    왜 필요한가:
        지금까지는 segment_by_gaps()로 학습 시점에 메모리에서만 결측을 잘라냈는데,
        이번 주 상황보고서용으로는 "실제로 얼마나 잘렸는지"를 파일로 남겨서 검토/공유할
        수 있어야 한다. 그래서 같은 절단 기준(max_gap_for_interp_min)을 원본 CSV
        전체(모든 컬럼)에 적용해 디스크에 저장하는 버전을 별도로 만들었다.

    절단 기준 (segment_by_gaps와 동일한 원칙을 원본 CSV 레벨로 옮긴 것):
        - HR/Calories (Activity) 두 컬럼만 기준으로 결측 구간을 찾는다
          (Libre GL 등 다른 컬럼은 결측이 없으므로 절단 사유가 되지 않음).
        - max_gap_for_interp_min분 이하로 짧게 끊긴 결측은 HR/Calories 두 컬럼만
          선형보간으로 메운다 (Meal Type, Carbs 같은 이벤트성 컬럼은 원본 그대로 둠 -
          그 컬럼들의 NaN은애초에 "결측"이 아니라 "그 시각에 식사가 없었다"는 정상 값이므로).
        - 그 값을 넘는 긴 결측 구간은 해당 시각의 행 자체를 통째로 삭제한다.
          Timestamp가 이어지지 않는 지점 = 원래 결측이 있었던 절단 지점이라는 뜻이므로,
          이후 이 파일을 읽는 쪽에서 Timestamp 간격만 보면 절단 지점을 그대로 알 수 있다
          (별도의 세그먼트 ID 컬럼을 만들 필요가 없음).

    Args:
        csv_path: 원본 환자 CSV 경로
        out_path: 절단된 복사본을 저장할 경로
        max_gap_for_interp_min: segment_by_gaps()와 동일한 의미의 임계값(분)

    Returns:
        dict(원본행수, 절단후행수, 유지비율(%)) 또는, Calories (Activity) 컬럼 자체가
        없어(예: CGMacros-007) 절단해도 애초에 쓸 수 없는 환자면 None을 반환하고
        파일도 만들지 않는다.
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Calories (Activity)" not in df.columns:
        # 이 컬럼이 아예 없으면 절단으로 해결될 문제가 아니라(보간할 원본 값 자체가 없음)
        # 복사본을 만들어봐야 의미가 없다 -> 생성하지 않고 명시적으로 알림
        return None

    # 주의: HR/Calories뿐 아니라 Libre GL(glucose, 타깃)도 반드시 포함해야 한다.
    # 대부분 환자는 Libre GL에 결측이 없지만(CGMacros-004 같은 예외 존재, 220분치 결측),
    # 여기서 glucose를 빼면 타깃에 NaN이 남은 채로 절단본이 만들어지고, 그 NaN이 그대로
    # 학습 타깃으로 들어가 손실이 첫 epoch부터 NaN이 되는 심각한 문제가 생긴다.
    # (raw 파이프라인의 segment_by_gaps()는 애초에 3개 피처 전체를 기준으로 검사해서
    #  이 문제가 없었음 - 여기서도 동일한 기준으로 맞춘다.)
    gap_cols = ["Libre GL", "HR", "Calories (Activity)"]
    mask = df[gap_cols].isna().any(axis=1).to_numpy()
    runs = _find_nan_runs(mask)

    short_run_mask = np.zeros(len(mask), dtype=bool)
    long_run_mask = np.zeros(len(mask), dtype=bool)
    for start, end in runs:
        if (end - start) <= max_gap_for_interp_min:
            short_run_mask[start:end] = True
        else:
            long_run_mask[start:end] = True

    interpolated = df[gap_cols].interpolate(method="linear", limit_direction="both")
    df.loc[short_run_mask, gap_cols] = interpolated.loc[short_run_mask, gap_cols]

    n_before = len(df)
    df_trimmed = df.loc[~long_run_mask].reset_index(drop=True)
    n_after = len(df_trimmed)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_trimmed.to_csv(out_path, index=False)

    return {
        "원본행수": n_before,
        "절단후행수": n_after,
        "유지비율(%)": round(n_after / n_before * 100, 1) if n_before else 0.0,
    }


def _concat_or_empty(arrays, fallback_shape):
    """리스트가 비어있으면 빈 배열을, 아니면 concatenate 결과를 반환 (환자가 0명 등 극단 케이스 방어)."""
    if len(arrays) == 0:
        return np.zeros(fallback_shape)
    return np.concatenate(arrays, axis=0)


def prepare_dataset_cgmacros(data_dir, feature_names, raw_column_names, column_map,
                              L, H, patient_limit=None, max_gap_for_interp_min=15):
    """
    전체 CGMacros 환자를 로드 -> 결측 기준으로 구간 분리 -> 각 구간 안에서만
    슬라이딩 윈도우 생성 -> train/val/test로 나눠 전체 환자에 대해 concat한다.

    data.py의 prepare_dataset()과 구조는 동일하되(환자 루프 -> 분할 -> 윈도우 -> concat),
    "환자 1명 = CSV 1개 = 항상 연속" 가정이 깨지므로 "환자 1명 = 여러 개의 연속 구간"으로
    한 단계 더 들어간 이중 루프 구조다.
    """
    splits = {"train": ([], []), "val": ([], []), "test": ([], []), "train_val": ([], [])}

    patients = discover_patients(data_dir, patient_limit)
    for pid, csv_path in patients:
        feature_df = load_single_patient_cgmacros(csv_path, feature_names, raw_column_names)
        segments = segment_by_gaps(feature_df, max_gap_for_interp_min)

        if not segments:
            print(f"  [건너뜀] {pid}: 사용 가능한 연속 구간이 없음 (필수 피처가 거의 전부 결측)")
            continue

        for data_array in segments:
            if len(data_array) < L + H:
                # 이 구간은 입력창(L)+예측시점(H)조차 못 채우는 너무 짧은 구간 -> 스킵
                continue

            train_data, val_data, test_data, train_val_data = split_data(data_array)

            for key, split_arr in (("train", train_data), ("val", val_data),
                                    ("test", test_data), ("train_val", train_val_data)):
                if len(split_arr) < L + H:
                    continue
                X, Y = extract_features(split_arr, feature_names, column_map)
                Xw, Yw = make_windows(X, Y, L, H)
                if len(Xw) == 0:
                    continue
                splits[key][0].append(Xw)
                splits[key][1].append(Yw)

    n_features = len(feature_names)
    result = {}
    for key, (xs, ys) in splits.items():
        X_cat = _concat_or_empty(xs, (0, L, n_features))
        Y_cat = _concat_or_empty(ys, (0, H))
        result[key] = (X_cat, Y_cat)
        print(f"  [{key}] 윈도우 {X_cat.shape[0]}개 생성")

    return result


def compute_means_variances_cgmacros(data_dir, feature_names, raw_column_names, column_map,
                                      L, H, patient_limit=None, max_gap_for_interp_min=15):
    """
    정규화(z-score)에 쓸 평균/표준편차를 train 구간에서만 계산한다.

    data.py의 compute_means_variances()와 동일한 역할이지만, prepare_dataset_cgmacros()와
    마찬가지로 "구간 분리"를 거친 뒤의 train 부분만 모아서 통계를 낸다.
    val/test 정보가 통계에 섞이면 안 되므로(데이터 누출 방지) train만 사용하는 원칙은
    원본 파이프라인과 동일하게 유지했다.
    """
    patients = discover_patients(data_dir, patient_limit)
    train_X_list = []

    for pid, csv_path in patients:
        feature_df = load_single_patient_cgmacros(csv_path, feature_names, raw_column_names)
        segments = segment_by_gaps(feature_df, max_gap_for_interp_min)
        for data_array in segments:
            if len(data_array) < L + H:
                continue
            train_data, _, _, _ = split_data(data_array)
            if len(train_data) < L + H:
                continue
            X_train, _ = extract_features(train_data, feature_names, column_map)
            train_X_list.append(X_train)

    if not train_X_list:
        raise ValueError("정규화 통계를 계산할 train 데이터가 없습니다 (환자/구간 필터를 확인하세요).")

    trainingset = np.concatenate(train_X_list, axis=0)
    mu_gen = np.mean(trainingset, axis=0)
    sigma_gen = np.std(trainingset, axis=0)
    mu_g = mu_gen[0]      # feature_names[0] == "glucose" 관례를 그대로 따름
    sigma_g = sigma_gen[0]

    return mu_g, sigma_g, mu_gen, sigma_gen


# =====================================================================
# 아래부터는 prepare_data_revision.py가 만든 "결측 절단 복사본"
# (data_revision/CGMacros/...)을 읽기 위한 함수들이다.
#
# 왜 위의 load_single_patient_cgmacros/segment_by_gaps를 그대로 못 쓰는가:
#   build_trimmed_copy()는 긴 결측 구간을 "행 자체를 삭제"하는 방식으로 잘라냈다.
#   그 결과 절단본 CSV에는 더 이상 NaN이 없다 (HR/Calories 둘 다 0건). 그런데
#   segment_by_gaps()는 NaN 패턴을 보고 절단 지점을 찾으므로, 이 파일을 그대로
#   넣으면 NaN이 하나도 없으니 "전체가 한 덩어리의 연속 구간"이라고 잘못 판단한다.
#   실제로는 행이 통째로 빠져 있어서 위/아래 행이 몇십 분씩 떨어져 있을 수 있는데,
#   make_windows()는 순전히 배열 위치(t-L:t)만 보고 슬라이딩하기 때문에 이 상태로
#   윈도우를 만들면 서로 몇 시간 떨어진 시점을 하나의 입력창으로 이어붙이는
#   심각한 오류가 생긴다. 그래서 NaN 대신 "Timestamp 간격이 원래 샘플링 간격(1분)보다
#   크다"는 조건으로 절단 지점을 다시 찾아야 한다.
def load_single_patient_revision(csv_path, feature_names, raw_column_names):
    """
    절단 복사본 CSV 1개를 읽어 Timestamp를 포함한 피처 DataFrame으로 반환한다.
    (원본용 load_single_patient_cgmacros와 달리 Timestamp를 버리지 않고 들고 있어야
    아래 segment_trimmed_by_timestamp()에서 절단 지점을 다시 찾을 수 있다.)
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    cols = {feat: df[raw_column_names[feat]].astype(float) for feat in feature_names}
    feature_df = pd.DataFrame(cols)[feature_names]
    feature_df.insert(0, "Timestamp", df["Timestamp"])
    return feature_df


def segment_trimmed_by_timestamp(feature_df_with_ts, normal_interval_min=1):
    """
    절단 복사본에서 실제로 시간이 이어지는 구간만 골라 세그먼트로 나눈다.

    로직: 연속된 두 행의 Timestamp 차이가 원래 샘플링 간격(1분)보다 크면,
    그 사이의 데이터가 삭제됐다는 뜻이므로 그 지점을 절단 지점으로 본다.
    build_trimmed_copy()가 지웠던 바로 그 자리를 Timestamp로 역추적하는 것.

    Args:
        feature_df_with_ts: load_single_patient_revision()의 반환값 (Timestamp 포함)
        normal_interval_min: 정상 샘플링 간격(분). CGMacros는 1분 간격 그리드이므로 기본값 1.

    Returns:
        [np.ndarray, ...] : segment_by_gaps()와 동일한 형태(피처 컬럼만, Timestamp 제외)의
                             연속 구간 리스트
    """
    ts = feature_df_with_ts["Timestamp"]
    gap_min = ts.diff().dt.total_seconds() / 60
    is_cut_point = (gap_min > normal_interval_min).fillna(False).to_numpy()

    seg_id = np.cumsum(is_cut_point)
    feature_only = feature_df_with_ts.drop(columns="Timestamp")
    segments = [g.to_numpy(dtype=float) for _, g in feature_only.groupby(seg_id)]
    return segments


def prepare_dataset_cgmacros_revision(data_dir, feature_names, raw_column_names, column_map,
                                       L, H, patient_limit=None):
    """
    data_revision(절단 복사본)으로부터 train/val/test 윈도우를 만든다.
    prepare_dataset_cgmacros()와 구조는 동일하고, "결측 탐지 방식"만 다르다
    (원본: NaN 패턴 / 절단본: Timestamp 간격) - 위 모듈 설명 참고.
    """
    splits = {"train": ([], []), "val": ([], []), "test": ([], []), "train_val": ([], [])}

    patients = discover_patients(data_dir, patient_limit)
    for pid, csv_path in patients:
        feature_df = load_single_patient_revision(csv_path, feature_names, raw_column_names)
        segments = segment_trimmed_by_timestamp(feature_df)

        if not segments:
            print(f"  [건너뜀] {pid}: 사용 가능한 연속 구간이 없음")
            continue

        for data_array in segments:
            if len(data_array) < L + H:
                continue

            train_data, val_data, test_data, train_val_data = split_data(data_array)

            for key, split_arr in (("train", train_data), ("val", val_data),
                                    ("test", test_data), ("train_val", train_val_data)):
                if len(split_arr) < L + H:
                    continue
                X, Y = extract_features(split_arr, feature_names, column_map)
                Xw, Yw = make_windows(X, Y, L, H)
                if len(Xw) == 0:
                    continue
                splits[key][0].append(Xw)
                splits[key][1].append(Yw)

    n_features = len(feature_names)
    result = {}
    for key, (xs, ys) in splits.items():
        X_cat = _concat_or_empty(xs, (0, L, n_features))
        Y_cat = _concat_or_empty(ys, (0, H))
        result[key] = (X_cat, Y_cat)
        print(f"  [{key}] 윈도우 {X_cat.shape[0]}개 생성")

    return result


def compute_means_variances_cgmacros_revision(data_dir, feature_names, raw_column_names, column_map,
                                               L, H, patient_limit=None):
    """정규화 통계를 절단 복사본의 train 구간에서만 계산한다 (원본용 함수와 동일한 원칙)."""
    patients = discover_patients(data_dir, patient_limit)
    train_X_list = []

    for pid, csv_path in patients:
        feature_df = load_single_patient_revision(csv_path, feature_names, raw_column_names)
        segments = segment_trimmed_by_timestamp(feature_df)
        for data_array in segments:
            if len(data_array) < L + H:
                continue
            train_data, _, _, _ = split_data(data_array)
            if len(train_data) < L + H:
                continue
            X_train, _ = extract_features(train_data, feature_names, column_map)
            train_X_list.append(X_train)

    if not train_X_list:
        raise ValueError("정규화 통계를 계산할 train 데이터가 없습니다 (환자/구간 필터를 확인하세요).")

    trainingset = np.concatenate(train_X_list, axis=0)
    mu_gen = np.mean(trainingset, axis=0)
    sigma_gen = np.std(trainingset, axis=0)
    mu_g = mu_gen[0]
    sigma_g = sigma_gen[0]

    return mu_g, sigma_g, mu_gen, sigma_gen
