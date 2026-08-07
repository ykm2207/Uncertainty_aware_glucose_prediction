#!/bin/bash
# run_ma_sweep.sh
# 8/7 이동평균(MA) 정책 재설계 이후 실험: MA 윈도우 30/60/180분 세 후보를
# horizon(30분/60분) 각각에 대해 전부 300epoch 정식으로 재학습한다.
#
# 이전(run_remaining_experiments.sh)에는 MA 적용 여부(True/False) + horizon마다
# python을 따로 실행했지만, 이번엔 "이동평균 값을 바꿔가며 한 번의 학습 실행 안에서
# 처리해달라"는 요청에 맞춰 MA 30/60/180분 스윕 자체는 train_cgmacros_revision.py
# 내부 for문이 처리한다(configs_cgmacros.py의 MA_WINDOW_CANDIDATES_MINUTES 참고).
# 그래서 이 스크립트는 horizon만 sed로 바꿔가며 train/evaluate를 한 번씩 실행하면 된다.
#
# 이동평균 미적용(0) 베이스라인은 이미 완료된 기존 결과(cgmacros_revision_h30/h60,
# 각각 300epoch 정식 학습분)를 그대로 재사용하므로 이번 스윕 대상이 아니다.
#
# 각 horizon 실행이 끝날 때마다 바로 git commit/push해서, 중간에 하나가 실패해도
# 그 앞까지의 결과는 이미 안전하게 GitHub에 올라가 있도록 했다.
set -e

# 이 스크립트가 어디서 실행되든 항상 레포 루트로 이동 (상대경로 DATA_DIR이
# 깨지지 않도록 - 지금까지 여러 번 겪었던 "경로 잘못돼서 데이터 0개" 문제 방지)
cd "$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_FILE="cgmacros/configs_cgmacros.py"

set_horizon() {
    local horizon_value="$1"
    sed -i "s/^HORIZON_MINUTES = .*/HORIZON_MINUTES = ${horizon_value}  # AUTO-SET by run_ma_sweep.sh/" "$CONFIG_FILE"
}

run_horizon() {
    local horizon_value="$1"
    echo ""
    echo "=============================================="
    echo "CGMacros MA 스윕 실행: horizon=${horizon_value}분 (MA 30/60/180분 전부, 한 번의 실행 안에서 순회)"
    echo "=============================================="
    set_horizon "$horizon_value"

    python cgmacros/train_cgmacros_revision.py
    python cgmacros/evaluate_cgmacros_revision.py

    git add cgmacros/configs_cgmacros.py \
        cgmacros_revision_ma*_tem_model.pth \
        cgmacros_revision_ma*_training_loss_plot.png \
        cgmacros_revision_ma*_ecp_graph_plot.png
    git commit -m "CGMacros MA 윈도우 스윕 학습결과 (horizon=${horizon_value}분, MA=30/60/180분)" || echo "(변경사항 없어 커밋 스킵)"
    git push

    echo "=== 완료: horizon=${horizon_value}분 (MA 30/60/180분) ==="
}

run_horizon 30
run_horizon 60

echo ""
echo "=============================================="
echo "CGMacros MA 윈도우 스윕 전체 완료 (2 horizon x 3 MA윈도우 = 6개 조합)"
echo "=============================================="
