#!/bin/bash
# run_ma_sweep.sh
# cgmacros/run_ma_sweep.sh와 동일한 이유/구조 (그쪽 상단 주석 참고).
# 8/7 이동평균(MA) 정책 재설계 이후 실험: MA 윈도우 30/60/180분 세 후보를
# horizon(30분/60분) 각각에 대해 전부 300epoch 정식으로 재학습한다.
# MA 스윕 자체는 train_shanghai.py 내부 for문이 한 번의 실행으로 처리하므로
# (configs_shanghai.py의 MA_WINDOW_CANDIDATES_MINUTES 참고), 이 스크립트는
# horizon만 바꿔가며 train/evaluate를 실행 + 커밋/푸시하면 된다.
# 이동평균 미적용(0) 베이스라인은 이미 완료된 기존 결과(shanghai_h30/h60)를 그대로
# 쓰므로 이번 스윕 대상이 아니다.
set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_FILE="shanghai/configs_shanghai.py"

set_horizon() {
    local horizon_value="$1"
    sed -i "s/^HORIZON_MINUTES = .*/HORIZON_MINUTES = ${horizon_value}  # AUTO-SET by run_ma_sweep.sh/" "$CONFIG_FILE"
}

run_horizon() {
    local horizon_value="$1"
    echo ""
    echo "=============================================="
    echo "Shanghai MA 스윕 실행: horizon=${horizon_value}분 (MA 30/60/180분 전부, 한 번의 실행 안에서 순회)"
    echo "=============================================="
    set_horizon "$horizon_value"

    python shanghai/train_shanghai.py
    python shanghai/evaluate_shanghai.py

    git add shanghai/configs_shanghai.py \
        shanghai_ma*_tem_model.pth \
        shanghai_ma*_training_loss_plot.png \
        shanghai_ma*_ecp_graph_plot.png
    git commit -m "Shanghai MA 윈도우 스윕 학습결과 (horizon=${horizon_value}분, MA=30/60/180분)" || echo "(변경사항 없어 커밋 스킵)"
    git push

    echo "=== 완료: horizon=${horizon_value}분 (MA 30/60/180분) ==="
}

run_horizon 30
run_horizon 60

echo ""
echo "=============================================="
echo "Shanghai MA 윈도우 스윕 전체 완료 (2 horizon x 3 MA윈도우 = 6개 조합)"
echo "=============================================="
