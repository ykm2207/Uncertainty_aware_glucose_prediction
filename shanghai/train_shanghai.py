# train_shanghai.py
# Shanghai_T2DM 데이터셋으로 TEM(evidential Transformer)을 학습하는 스크립트.
# 구조는 train_cgmacros.py와 동일하다 (train_evidential_model 학습 루프를 그대로 복제해서
# 사용하는 이유는 train_cgmacros.py 상단 주석 참고 - train.py를 직접 import하면 안 됨).
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# shanghai/ 폴더에서도 루트의 공용 data.py/model.py/utils.py를 그대로 쓰기 위해
# 루트 디렉터리를 sys.path에 추가 (repo 루트에서 실행하는 걸 기준으로 함).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import normalize_features, BGDataset
from data_shanghai import prepare_dataset_shanghai, compute_means_variances_shanghai
from utils import evidential_data_loss, kl_reg_loss_term, amini_reg_loss_term, get_device
from model import e_Transformers
from configs_shanghai import (
    DATA_DIR, FEATURES, COLUMN_MAP,
    INPUT_TIMESTEPS, HORIZON_LENGTH, MIN_SESSION_ROWS, SESSION_LIMIT,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LAMBDA_REG, REG_TERM, DEVICE,
    D_MODEL, N_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN,
    MODEL_SAVE_PATH, LOSS_SAVE_PATH,
)


def train_evidential_model(
    model, train_loader, val_loader, device,
    num_epochs=300, lr=1e-4, lambda_reg=0.01, reg_term="kl",
    scheduler_lambda=None, loss_save_path=None, model_save_path=None,
):
    """train_cgmacros.py의 동일 함수와 완전히 같은 학습 루프 (원 논문 저자 코드 그대로)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=scheduler_lambda if scheduler_lambda else (lambda e: 1.0)
    )

    training_loss = []
    validation_loss = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            L_p_alpha, L_p_beta, L_p_nu, L_pred = model(X_batch)
            loss = evidential_data_loss(L_pred, y_batch, L_p_nu, L_p_alpha, L_p_beta)

            if reg_term == "kl":
                reg_loss = kl_reg_loss_term(y_batch, L_pred, L_p_alpha, L_p_beta)
            else:
                reg_loss = amini_reg_loss_term(y_batch, L_pred, L_p_nu, L_p_alpha)

            total_loss = loss + lambda_reg * reg_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        training_loss.append(train_loss)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                L_v_alpha, L_v_beta, L_v_nu, L_v_pred = model(X_val)
                v_loss = evidential_data_loss(L_v_pred, y_val, L_v_nu, L_v_alpha, L_v_beta)

                if reg_term == "kl":
                    vreg_loss = kl_reg_loss_term(y_val, L_v_pred, L_v_alpha, L_v_beta)
                else:
                    vreg_loss = amini_reg_loss_term(y_val, L_v_pred, L_v_nu, L_v_alpha)

                total_v_loss = v_loss + lambda_reg * vreg_loss
                val_loss += total_v_loss.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        validation_loss.append(val_loss)

        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(training_loss, label="Training Loss")
    axes[0].plot(validation_loss, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"N_param = {n_p:.3e}", fontsize=9)
    axes[0].legend()

    if len(validation_loss) > 1:
        rel_tol = [np.abs((validation_loss[k] - validation_loss[k + 1]) / validation_loss[k])
                   for k in range(len(validation_loss) - 1)]
        m_rel = np.median(rel_tol[-10:]) if len(rel_tol) >= 10 else np.median(rel_tol)
        axes[1].plot(rel_tol[max(0, len(rel_tol) - 50):])
        axes[1].set_title(f"Rel. Tol. Val., median of last 10 = {m_rel:.3e}", fontsize=9)
    else:
        m_rel = float("nan")

    plt.tight_layout()
    if loss_save_path:
        plt.savefig(loss_save_path)
    plt.close(fig)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    return model, training_loss, validation_loss, m_rel


def make_loader(X, Y, batch_size, shuffle=False):
    return DataLoader(BGDataset(X, Y), batch_size=batch_size, shuffle=shuffle)


def main():
    print("=== Shanghai_T2DM 데이터 로딩 ===")
    data_splits = prepare_dataset_shanghai(
        DATA_DIR, FEATURES, COLUMN_MAP, L=INPUT_TIMESTEPS, H=HORIZON_LENGTH,
        session_limit=SESSION_LIMIT, min_session_rows=MIN_SESSION_ROWS,
    )
    mu_g, sigma_g, mu_gen, sigma_gen = compute_means_variances_shanghai(
        DATA_DIR, FEATURES, COLUMN_MAP, L=INPUT_TIMESTEPS, H=HORIZON_LENGTH,
        session_limit=SESSION_LIMIT, min_session_rows=MIN_SESSION_ROWS,
    )

    data_norm = {}
    for key, (X, Y) in data_splits.items():
        X_n = normalize_features(X, mu_gen, sigma_gen)
        Y_n = normalize_features(Y, np.array([mu_g]), np.array([sigma_g]))
        data_norm[key] = (X_n, Y_n)

    val_input_n, val_output_n = data_norm["val"]
    train_val_input_n, train_val_output_n = data_norm["train_val"]

    val_loader = make_loader(val_input_n, val_output_n, BATCH_SIZE)
    train_val_loader = make_loader(train_val_input_n, train_val_output_n, BATCH_SIZE, shuffle=True)

    device = get_device(preferred=DEVICE)

    model = e_Transformers(
        input_dim=len(FEATURES), d_model=D_MODEL, n_heads=N_HEADS,
        num_layers=NUM_LAYERS, ff_dim=FF_DIM, output_dim=HORIZON_LENGTH, max_len=MAX_LEN,
    )

    print("=== 학습 시작 ===")
    train_evidential_model(
        model,
        train_loader=train_val_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        reg_term=REG_TERM,
        scheduler_lambda=lambda e: 1.0 if e < NUM_EPOCHS * 0.75 else 0.1,
        loss_save_path=LOSS_SAVE_PATH,
        model_save_path=MODEL_SAVE_PATH,
    )
    print(f"모델 저장 완료: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
