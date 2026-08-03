# evaluate_cgmacros.py
# CGMacros로 학습한 TEM 모델을 테스트셋으로 평가한다.
#
# 이번 실험 조건은 RMSE/MAE를 1차 지표로 요구했으므로 그것을 가장 먼저 출력하고,
# 원 논문이 쓰던 MARD/DTS 존/Brier/AUC/ECP도 참고용으로 이어서 계산한다.
# 원 논문 지표들은 "저혈당/고혈당 위험구간을 얼마나 잘 잡아내는가", "예측 불확실성이
# 실제 오차와 얼마나 잘 맞아떨어지는가(calibration)"까지 보는데, RMSE/MAE는 점 예측의
# 평균적인 크기 오차만 본다. 즉 RMSE/MAE가 낮아도 저혈당을 못 잡아낼 수 있고,
# 반대로 원 논문 지표가 나빠도 RMSE/MAE는 괜찮아 보일 수 있다 -> 이 차이를 evaluate 끝에
# 명시적으로 출력해서 "지표를 바꿨을 때 뭐가 달라지는지" 확인할 수 있게 했다.
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, t as student_t
import matplotlib.pyplot as plt

# cgmacros/ 폴더에서도 루트의 공용 data.py/model.py/utils.py를 그대로 쓰기 위해
# 루트 디렉터리를 sys.path에 추가 (repo 루트에서 실행하는 걸 기준으로 함).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import normalize_features, BGDataset
from data_cgmacros import prepare_dataset_cgmacros, compute_means_variances_cgmacros
from utils import (rmse, mae, get_device, sensitivity_metric, CI_calculation, DTS_error_zone_count,
                    f_auc_hypo_score, f_auc_hyper_score, f_brier_hypo_score, f_brier_hyper_score)
from model import e_Transformers
from configs_cgmacros import (
    DATA_DIR, FEATURES, RAW_COLUMN_NAMES, COLUMN_MAP,
    INPUT_TIMESTEPS, HORIZON_LENGTH, MAX_GAP_FOR_INTERP_MIN, PATIENT_LIMIT,
    BATCH_SIZE, DEVICE, D_MODEL, N_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN,
    MODEL_SAVE_PATH, ECP_GRAPH_SAVE_PATH,
)


def evaluate_model_evidential(model, loader, sigma_g, mu_g, ecp_graph_save_path):
    """
    evaluate.py의 동일 함수와 로직은 같되, 맨 앞에 RMSE/MAE를 추가로 계산해서
    metrics_dict에 넣는다. 저혈당/고혈당 샘플이 0건인 경우(작은 patient_limit로
    스모크 테스트 할 때 흔함) 관련 지표 계산에서 나눗셈 경고/NaN이 나지 않도록
    "정의 불가"로 명시적으로 표시하고 건너뛴다.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_targets, all_alpha, all_beta, all_nu = [], [], [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            alpha, beta, nu, preds = model(x_batch)

            all_preds.append(preds.cpu())
            all_alpha.append(alpha.cpu())
            all_beta.append(beta.cpu())
            all_nu.append(nu.cpu())
            all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_alpha = torch.cat(all_alpha).numpy()
    all_beta = torch.cat(all_beta).numpy()
    all_nu = torch.cat(all_nu).numpy()
    all_targets = torch.cat(all_targets).numpy()

    all_preds_real = (all_preds * sigma_g + mu_g).reshape(-1)
    all_beta_real = (all_beta * sigma_g * sigma_g).reshape(-1)
    all_alpha_real = all_alpha.reshape(-1)
    all_nu_real = all_nu.reshape(-1)
    all_targets_real = (all_targets * sigma_g + mu_g).reshape(-1)
    t_std = np.sqrt(all_beta_real * (1 + all_nu_real) / (all_alpha_real * all_nu_real))

    metrics_dict = {
        "RMSE": rmse(all_targets_real, all_preds_real),
        "MAE": mae(all_targets_real, all_preds_real),
    }

    # ===== 원래 쓰던 지표(원 논문 기준) 계산 — 다시 활성화함 =====
    metrics_dict["MARD"] = np.mean(np.abs(all_preds_real - all_targets_real) / all_targets_real) * 100

    arr = DTS_error_zone_count(all_targets_real, all_preds_real)
    counts = np.bincount(arr, minlength=5)
    percentages = 100 * counts / counts.sum()
    for i, zone in enumerate("ABCDE"):
        metrics_dict[f"Zone {zone} accuracy"] = percentages[i]

    if np.std(t_std) > 0 and np.std(arr) > 0:
        arr_rho, _ = spearmanr(t_std, arr)
        metrics_dict["Correlation with risk zones"] = arr_rho
    else:
        metrics_dict["Correlation with risk zones"] = float("nan")

    Llower = all_preds_real + CI_calculation(0.68, all_alpha_real, all_beta_real, all_nu_real, all_preds_real)
    Lupper = all_preds_real - CI_calculation(0.68, all_alpha_real, all_beta_real, all_nu_real, all_preds_real)

    target_level_70_hypo = (all_targets_real < 70).astype(int)
    target_level_180_hyper = (all_targets_real > 180).astype(int)

    if target_level_70_hypo.sum() > 0:
        pred_level_70_hypo = (Llower < 70).astype(int)
        metrics_dict["Level 70 Sensitivity"] = sensitivity_metric(target_level_70_hypo, pred_level_70_hypo)
        metrics_dict["Brier_Hypo"] = f_brier_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
        metrics_dict["AUC_Hypo"] = f_auc_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
    else:
        print("  [참고] 테스트셋에 저혈당(<70) 샘플이 없어 Hypo 관련 지표는 '정의 불가'로 표시")
        metrics_dict["Level 70 Sensitivity"] = float("nan")
        metrics_dict["Brier_Hypo"] = float("nan")
        metrics_dict["AUC_Hypo"] = float("nan")

    if target_level_180_hyper.sum() > 0:
        pred_level_180_hyper = (Lupper > 180).astype(int)
        metrics_dict["Level 180 Sensitivity"] = sensitivity_metric(target_level_180_hyper, pred_level_180_hyper)
        metrics_dict["Brier_Hyper"] = f_brier_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
        metrics_dict["AUC_Hyper"] = f_auc_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
    else:
        print("  [참고] 테스트셋에 고혈당(>180) 샘플이 없어 Hyper 관련 지표는 '정의 불가'로 표시")
        metrics_dict["Level 180 Sensitivity"] = float("nan")
        metrics_dict["Brier_Hyper"] = float("nan")
        metrics_dict["AUC_Hyper"] = float("nan")

    pred_std = t_std
    pred_error = np.abs(all_preds_real - all_targets_real)
    rho_uncertainty, _ = spearmanr(pred_std.flatten(), pred_error.flatten())
    metrics_dict["Correlation with error"] = rho_uncertainty

    ecp_list, dis_list = [], []
    for j in range(99):
        interval_CI = (j + 1) * 0.01
        neg_CI_delta = student_t.interval(interval_CI, 2 * all_alpha_real, loc=0, scale=pred_std)[1]
        acc = np.sum(np.abs(all_targets_real - all_preds_real) < np.abs(neg_CI_delta))
        empirical_ecp = acc / len(all_targets_real)
        discrepancy = np.abs(empirical_ecp - interval_CI)
        ecp_list.append(empirical_ecp)
        dis_list.append(discrepancy)
    ecp_list.append(1.0)
    metrics_dict["MCE"] = np.mean(dis_list)

    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(0.01, 1.01, 0.01), ecp_list, label=f'ECP (MCE = {metrics_dict["MCE"]:.2e})')
    plt.plot([0, 1], [0, 1], "r--", label="Nominal Coverage")
    plt.xlabel("Nominal coverage prob.")
    plt.ylabel("Empirical coverage prob.")
    plt.legend(fontsize=9)
    plt.title("Error Calibration Plot (CGMacros)", fontsize=10)
    plt.tight_layout()
    plt.savefig(ecp_graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metrics_dict


def make_loader(X, Y, batch_size, shuffle=False):
    return DataLoader(BGDataset(X, Y), batch_size=batch_size, shuffle=shuffle)


def main():
    print("=== CGMacros 데이터 로딩 ===")
    data_splits = prepare_dataset_cgmacros(
        DATA_DIR, FEATURES, RAW_COLUMN_NAMES, COLUMN_MAP,
        L=INPUT_TIMESTEPS, H=HORIZON_LENGTH,
        patient_limit=PATIENT_LIMIT, max_gap_for_interp_min=MAX_GAP_FOR_INTERP_MIN,
    )
    mu_g, sigma_g, mu_gen, sigma_gen = compute_means_variances_cgmacros(
        DATA_DIR, FEATURES, RAW_COLUMN_NAMES, COLUMN_MAP,
        L=INPUT_TIMESTEPS, H=HORIZON_LENGTH,
        patient_limit=PATIENT_LIMIT, max_gap_for_interp_min=MAX_GAP_FOR_INTERP_MIN,
    )

    data_norm = {}
    for key, (X, Y) in data_splits.items():
        X_n = normalize_features(X, mu_gen, sigma_gen)
        Y_n = normalize_features(Y, np.array([mu_g]), np.array([sigma_g]))
        data_norm[key] = (X_n, Y_n)

    test_input_n, test_output_n = data_norm["test"]
    test_loader = make_loader(test_input_n, test_output_n, BATCH_SIZE)

    device = get_device(preferred=DEVICE)
    model = e_Transformers(
        input_dim=len(FEATURES), d_model=D_MODEL, n_heads=N_HEADS,
        num_layers=NUM_LAYERS, ff_dim=FF_DIM, output_dim=HORIZON_LENGTH, max_len=MAX_LEN,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    print("=== 평가 시작 ===")
    metrics = evaluate_model_evidential(model, test_loader, sigma_g, mu_g, ECP_GRAPH_SAVE_PATH)

    print("\n[CGMacros 평가 결과]")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}" if not np.isnan(v) else f"{k}: 정의 불가(해당 샘플 없음)")

    print(
        "\n[지표 변경에 따른 차이 안내]\n"
        "RMSE/MAE는 예측값과 실제값의 평균적인 크기 오차만 반영하는 지표입니다.\n"
        "원 논문(HUPA 기준)이 핵심으로 삼은 MARD/DTS 존 정확도/Brier/AUC/ECP(MCE)는\n"
        "저혈당·고혈당처럼 임상적으로 위험한 구간을 얼마나 잘 잡아내는지, 그리고\n"
        "모델이 내놓은 불확실성(alpha/beta/nu)이 실제 오차 분포와 얼마나 맞는지(calibration)까지 봅니다.\n"
        "따라서 RMSE/MAE만으로는 '위험 구간 탐지 성능'이나 '불확실성 신뢰도'는 알 수 없으므로,\n"
        "위에 같이 출력한 원 논문 지표도 함께 참고해야 합니다."
    )


if __name__ == "__main__":
    main()
