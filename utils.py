import torch
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import t as student_t


def rmse(y_true, y_pred):
    """
    RMSE(Root Mean Squared Error, 평균제곱근오차)를 계산한다.

    왜 필요한가:
        원 논문(HUPA-UCM 기준)은 MARD/DTS 존 정확도/Brier/AUC/ECP처럼
        저혈당·고혈당 위험구간과 불확실성 보정(calibration)까지 함께 보는
        임상 지표 위주였다. 하지만 CGMacros/Shanghai 재현 실험에서는 팀 지정에 따라
        "점 예측이 실제 혈당값에서 얼마나 벗어났는가"만 보는 RMSE/MAE를 1차 지표로 쓰기로 했다.
        RMSE는 큰 오차(스파이크 예측 실패 등)에 제곱으로 페널티를 줘서 MAE보다 민감하다.

    Args:
        y_true, y_pred : 정규화를 해제한(mg/dL 단위) 실제값/예측값 배열. shape은 동일해야 함.

    Returns:
        float: RMSE 값 (mg/dL 단위)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """
    MAE(Mean Absolute Error, 평균절대오차)를 계산한다.

    왜 필요한가:
        RMSE와 함께 보는 이유는, RMSE는 큰 오차(이상치)에 민감하게 반응하는 반면
        MAE는 오차 크기에 비례해서만 반응하므로 "평균적으로 몇 mg/dL 틀리는가"를
        더 직관적으로 보여준다. 두 지표를 같이 보고하면 예측 오차 분포에
        큰 이상치가 섞여 있는지(RMSE≫MAE)까지 함께 판단할 수 있다.

    Args:
        y_true, y_pred : 정규화를 해제한(mg/dL 단위) 실제값/예측값 배열. shape은 동일해야 함.

    Returns:
        float: MAE 값 (mg/dL 단위)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def get_device(preferred="mps"):
    """
    학습/평가에 사용할 torch device를 결정한다.

    왜 필요한가:
        기존 코드(train.py/evaluate.py)는 "cuda 있으면 cuda, 아니면 cpu"만 판단해서
        Apple Silicon(M1/M2 등) 맥북에서는 GPU 가속(MPS 백엔드)을 전혀 못 썼다.
        이번 실험은 맥북에서 먼저 파이프라인이 도는지 MPS로 확인하고,
        추후 랩 서버(CUDA)에서 본 학습을 돌릴 계획이므로 두 백엔드를 모두 지원하되
        기본 우선순위를 config에서 바꿀 수 있게 만들었다.

    Args:
        preferred: "mps" | "cuda" | "cpu" 중 우선적으로 시도할 백엔드.
                   해당 백엔드를 이 기기에서 못 쓰면 mps -> cuda -> cpu 순으로 자동 폴백한다.

    Returns:
        torch.device: 실제로 사용할 device. 선택된 device는 콘솔에 한글로 출력한다.
    """
    candidates = {
        "mps": lambda: torch.backends.mps.is_available(),
        "cuda": lambda: torch.cuda.is_available(),
        "cpu": lambda: True,
    }

    # 사용자가 지정한 backend를 맨 앞으로, 나머지는 mps -> cuda -> cpu 순으로 폴백 순서를 구성
    order = [preferred] + [b for b in ("mps", "cuda", "cpu") if b != preferred]

    for backend in order:
        if candidates[backend]():
            device = torch.device(backend)
            print(f"[device] '{backend}' 사용 (요청: '{preferred}')")
            return device

    # 이론상 cpu는 항상 True이므로 여기까지 오지 않지만, 방어적으로 폴백
    print("[device] 사용 가능한 backend를 찾지 못해 cpu로 폴백")
    return torch.device("cpu")


def kl_reg_loss_term(u_obs, gamma, alpha, beta):
    """
    Regularization function of Evidential Regression by H.S.Tan et al.
    """
    euler_constant = 0.5772156649015329
    beta_r = u_obs.max()**2
    kl_div = alpha*torch.log(beta_r/beta) + torch.lgamma(alpha) + \
              euler_constant*(alpha - 1) + (beta - beta_r)/beta_r

    return torch.mean(torch.abs(u_obs - gamma)*kl_div)


def amini_reg_loss_term(u_obs, gamma, nu, alpha):

    """
    Regularization function of Evidential Regression by Amini et al.
    """
    reg_term = torch.abs(u_obs - gamma)*(2*nu + alpha)

    return torch.mean(reg_term)


def evidential_data_loss(u_obs, gamma, nu, alpha, beta ):

    """
    Loss function of Evidential Regression

    Args:
        u_obs : observed target
        alpha, beta, nu, gamma : evidential model's outputs with 'gamma' being mean

    Returns:
        the main loss term of evidential regression.
    """

    twoBlambda = 2*(beta)*(1+nu)

    nll = 0.5*torch.log(torch.pi/(nu))  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(nu*(u_obs-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll)



def sensitivity_metric(y_true, y_pred):
    
    """ Computes sensitivity metric """

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    sensitivity = TP / (TP + FN + 1e-5)

    return sensitivity


def CI_calculation(interval_CI, alpha, beta, nu, gamma):

    """
    Function that computes the confidence interval for the model's output

    Args:
        interval_CI : confidence interval, e.g. 0.95 for 95% interval
        alpha, beta, nu, gamma : evidential model's outputs with 'gamma' being mean

    Returns:
        lower bound of the confidence interval assuming mean centered at 0
    """

    scale_sq = beta*(1+nu)/(alpha*nu)
    scale = np.sqrt(scale_sq)

    I_delta = []

    for i in range(len(alpha)):
      a,b= student_t.interval(interval_CI, df=2*alpha[i], loc=0, scale=scale[i])
      I_delta.append(a)

    return np.array(I_delta)



def f_auc_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  """
  Computes the auc score for evidential hypoglycemia prediction.

  Args:
      (i)all_preds_real: evidential mean
      (ii)all_beta_real: evidential beta
      (iii)all_nu_real: evidential nu
      (iv)all_alpha_real: evidential alpha
      (v)all_targets_real: targets

  Returns: PR-AUC score for evidential hypoglycemia prediction.

  """
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (70 - t_mean) / t_sigma
  p_hypo = student_t.cdf(z, df=deg_of_freedom)
  target_hypo = (all_targets_real <= 70).astype(int)
  auc = average_precision_score(target_hypo.flatten(), p_hypo.flatten())

  return auc



def f_auc_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  """
  Computes the auc score for evidential hyperglycemia prediction.

  Args:
      (i)all_preds_real: evidential mean
      (ii)all_beta_real: evidential beta
      (iii)all_nu_real: evidential nu
      (iv)all_alpha_real: evidential alpha
      (v)all_targets_real: targets

  Returns: PR-AUC score for evidential hyperglycemia prediction.
  """
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (180 - t_mean) / t_sigma
  p_lt_180 = student_t.cdf(z, df=deg_of_freedom)
  p_hyper = 1 - p_lt_180
  target_hyper = (all_targets_real > 180).astype(int)
  auc = average_precision_score(target_hyper.flatten(), p_hyper.flatten())

  return auc



def DTS_error_zone_count(act,pred):
  """
  This function outputs the DTS Error Grid region, based on the article
  https://pubmed.ncbi.nlm.nih.gov/39369312/

  Arguments:
  act : actual value
  pred : predicted value

  """

  def DTS_error_zone_detailed(act, pred):

    """
    Arguments:
    act : actual value
    pred : predicted value

    Returns:
    DTS Error Grid region: A(0), B(1), C(2), D(3), E(4)
    """

    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def dts_error(act, pred):
        # Zone A
        if (pred < 60 and act < 62.5) or (act > 50 and above_line(62.5, 50, 600, 480) \
                and below_line(50, 60, 500, 600)):
            return 0

        # Zone B
        if (pred < 86.5 and act < 97.5) or (act > 50 and above_line(97.5, 50, 600, 307) \
                and below_line(50, 86.5, 347, 600)):
            return 1

        # Zone C
        if (pred < 124 and act < 153) or (act > 50 and above_line(153, 50, 600, 197) \
                and below_line(50, 124, 241, 600)):
            return 2

        # Zone D
        if (pred < 179 and act < 238) or (act > 50 and above_line(238, 50, 600, 126) \
                and below_line(50, 179, 167, 600)):
            return 3

        # Zone E
        return 4

    return dts_error(act, pred)

  DTS_error_zone_detailed = np.vectorize(DTS_error_zone_detailed)
  zones = DTS_error_zone_detailed(act, pred)

  return zones


def DTS_zones(targets, preds):
  """
  Computes the DTS zone grid accuracies from zone A to E

  Args:
    targets : actual values
    preds : predicted values

  Returns: a list of zone accuracies from A to E
  """
  zones = DTS_error_zone_count(targets, preds)
  counts = np.bincount(zones, minlength=5)
  percentages = 100 * counts / counts.sum()

  return percentages



def f_brier_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  '''

  Computes the Brier score for hypoglycemia prediction.

  '''
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (70 - t_mean) / t_sigma
  p_no_hypo = 1 - student_t.cdf(z, df=deg_of_freedom)
  target_no_hypo = (all_targets_real > 70).astype(int)
  b_score = np.mean((target_no_hypo - p_no_hypo)**2)

  return b_score



def f_brier_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  '''

  Computes the Brier score for hyperglycemia prediction.

  '''
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (180 - t_mean) / t_sigma
  p_lt_180 = student_t.cdf(z, df=deg_of_freedom)
  p_no_hyper = p_lt_180
  target_no_hyper = (all_targets_real <= 180).astype(int)
  b_score = np.mean((target_no_hyper - p_no_hyper)**2)

  return b_score