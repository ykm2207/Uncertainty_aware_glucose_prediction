#!/bin/bash
# run_remaining_experiments.sh
# Shanghai에 남은 실험 조합(MA 적용/60분, MA 미적용/60분)을 순서대로 자동으로 돌린다.
# cgmacros/run_remaining_experiments.sh와 동일한 이유/구조 (그쪽 상단 주석 참고).
set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_FILE="shanghai/configs_shanghai.py"

set_config() {
    local ma_value="$1"      # True 또는 False
    local horizon_value="$2" # 30 또는 60
    sed -i "s/^APPLY_MOVING_AVERAGE = .*/APPLY_MOVING_AVERAGE = ${ma_value}/" "$CONFIG_FILE"
    sed -i "s/^HORIZON_MINUTES = .*/HORIZON_MINUTES = ${horizon_value}  # AUTO-SET by run_remaining_experiments.sh/" "$CONFIG_FILE"
}

run_combo() {
    local ma_value="$1"
    local horizon_value="$2"
    echo ""
    echo "=============================================="
    echo "Shanghai 실행: MA=${ma_value}, horizon=${horizon_value}분"
    echo "=============================================="
    set_config "$ma_value" "$horizon_value"

    python shanghai/train_shanghai.py
    python shanghai/evaluate_shanghai.py

    git add shanghai/configs_shanghai.py \
        shanghai*_tem_model.pth \
        shanghai*_training_loss_plot.png \
        shanghai*_ecp_graph_plot.png
    git commit -m "Shanghai 학습결과 (MA=${ma_value}, horizon=${horizon_value}분)" || echo "(변경사항 없어 커밋 스킵)"
    git push

    echo "=== 완료: MA=${ma_value}, horizon=${horizon_value}분 ==="
}

run_combo True 60
run_combo False 60

echo ""
echo "=============================================="
echo "Shanghai 남은 실험 조합 전부 완료"
echo "=============================================="
