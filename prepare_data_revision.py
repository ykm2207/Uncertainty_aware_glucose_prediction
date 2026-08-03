# prepare_data_revision.py
# CGMacros 원본 CSV들을 읽어서 HR/Calories(Activity)의 긴 결측 구간을 절단한
# 복사본을 ./data_revision/CGMacros/ 아래에 만든다.
#
# 왜 별도 스크립트로 뽑았나:
#   data_cgmacros.py의 segment_by_gaps()는 학습 시점에 "메모리에서만" 결측을 잘라내는
#   함수라 실행할 때마다 매번 다시 계산된다. 이번엔 실제로 잘려나간 결과를 파일로 남겨서
#   (1) 사수 상황보고서에 절단 전/후 수치를 그대로 보여주고 (2) 필요하면 이 복사본을
#   DATA_DIR로 지정해서 재사용할 수 있게 하기 위해 디스크에 저장하는 버전을 따로 뒀다.
#   절단 기준(HR/Calories 결측, 15분 임계값)은 configs_cgmacros.py와 완전히 동일하다.
import os
import pandas as pd

from configs_cgmacros import DATA_DIR, MAX_GAP_FOR_INTERP_MIN
from data_cgmacros import discover_patients, build_trimmed_copy

OUT_ROOT = "./data_revision/CGMacros"


def main():
    patients = discover_patients(DATA_DIR)
    rows = []

    print(f"=== CGMacros 결측 절단 복사본 생성 (max_gap_for_interp_min={MAX_GAP_FOR_INTERP_MIN}분) ===")
    for pid, csv_path in patients:
        out_path = os.path.join(OUT_ROOT, pid, f"{pid}.csv")
        result = build_trimmed_copy(csv_path, out_path, MAX_GAP_FOR_INTERP_MIN)

        if result is None:
            print(f"  [제외] {pid}: 'Calories (Activity)' 컬럼이 원본에 없어 절단 대상이 아님 (복사본 생성 안 함)")
            rows.append({"환자": pid, "원본행수": None, "절단후행수": 0, "유지비율(%)": 0.0, "비고": "필수 컬럼 없음"})
        else:
            print(f"  [완료] {pid}: {result['원본행수']}행 -> {result['절단후행수']}행 "
                  f"({result['유지비율(%)']}% 유지)")
            rows.append({"환자": pid, **result, "비고": ""})

    summary = pd.DataFrame(rows)
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUT_ROOT, "trim_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== 요약 (상황보고서용) ===")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summary.to_string(index=False))
    kept = summary[summary["원본행수"].notna()]
    print(f"\n복사본 생성된 환자: {len(kept)} / {len(patients)}명")
    print(f"평균 유지비율: {kept['유지비율(%)'].mean():.1f}%")
    print(f"요약 파일 저장: {summary_path}")


if __name__ == "__main__":
    main()
