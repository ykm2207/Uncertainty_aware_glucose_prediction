#!/bin/bash
# run_remaining_experiments.sh
# CGMacros에 남은 실험 조합(MA 미적용/30분, MA 미적용/60분, MA 적용/60분)을
# 순서대로 자동으로 돌린다. 사람이 매번 config 고치고 재실행할 필요 없이
# 서버에서 tmux 세션 안에 이 스크립트 하나만 실행해두면 끝까지 알아서 진행된다.
#
# 왜 이렇게 만들었나: 오늘 안에 결과를 다 만들어야 하는데, 조합마다 매번
# "config 수정 -> 학습 -> 평가 -> push"를 손으로 반복하면 시간도 오래 걸리고
# 실수(예: config 안 바꾸고 실행)하기도 쉽다. 그래서 조합별로 확실하게 config를
# 맞춘 뒤 학습+평가+커밋+푸시까지 한 번에 처리하는 스크립트로 만들었다.
#
# 각 실행이 끝날 때마다 바로 git commit/push를 해서, 중간에 하나가 실패해도
# 그 앞까지의 결과는 이미 안전하게 GitHub에 올라가 있도록 했다.
set -e

# 이 스크립트가 어디서 실행되든 항상 레포 루트로 이동 (상대경로 DATA_DIR이
# 깨지지 않도록 - 지금까지 여러 번 겪었던 "경로 잘못돼서 데이터 0개" 문제 방지)
cd "$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_FILE="cgmacros/configs_cgmacros.py"

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
    echo "CGMacros 실행: MA=${ma_value}, horizon=${horizon_value}분"
    echo "=============================================="
    set_config "$ma_value" "$horizon_value"

    python cgmacros/train_cgmacros_revision.py
    python cgmacros/evaluate_cgmacros_revision.py

    git add cgmacros/configs_cgmacros.py \
        cgmacros_revision*_tem_model.pth \
        cgmacros_revision*_training_loss_plot.png \
        cgmacros_revision*_ecp_graph_plot.png
    git commit -m "CGMacros 학습결과 (MA=${ma_value}, horizon=${horizon_value}분)" || echo "(변경사항 없어 커밋 스킵)"
    git push

    echo "=== 완료: MA=${ma_value}, horizon=${horizon_value}분 ==="
}

run_combo False 30
run_combo False 60
run_combo True 60

echo ""
echo "=============================================="
echo "CGMacros 남은 실험 조합 전부 완료"
echo "=============================================="
