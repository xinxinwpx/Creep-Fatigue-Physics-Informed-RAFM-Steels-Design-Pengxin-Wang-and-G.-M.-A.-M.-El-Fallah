# Creep–Fatigue Design of RAFM Steels Using Physics-Informed and Data-Driven Surrogate Models with Multi-Objective Optimisation
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
import torch
import torch.nn as nn
import numpy as np
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 1. Model Architecture (MUST match v12.6 exactly)
# ================================================================
class DeepBranch(nn.Module):
    def __init__(self, in_dim, hidden, depth, dropout):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(in_dim if i==0 else hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout)
            ]
        self.net = nn.Sequential(*layers)
        self.res = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)


class AttentionFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        m = min(dims)
        self.proj = nn.ModuleList([nn.Linear(d, m) for d in dims])
        self.W = nn.Linear(m*3, 3)
        self.proj_out = nn.Linear(m, m)

    def forward(self, f1, f2, f3):
        f1p, f2p, f3p = self.proj[0](f1), self.proj[1](f2), self.proj[2](f3)
        weights = torch.softmax(self.W(torch.cat([f1p, f2p, f3p], dim=1)), dim=1)
        fused = weights[:,0:1]*f1p + weights[:,1:2]*f2p + weights[:,2:3]*f3p
        return self.proj_out(fused)


class DeepHybridPINN(nn.Module):
    def __init__(self, dim1, dim2, dim3, hidden, depth, dropout):
        super().__init__()
        self.branch1 = DeepBranch(dim1, hidden, depth, dropout)
        self.branch2 = DeepBranch(dim2, hidden, depth, dropout)
        self.branch3 = DeepBranch(dim3, hidden, depth, dropout)
        self.fusion = AttentionFusion([hidden, hidden, hidden])

        self.out = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, x3):
        f1 = self.branch1(x1)
        f2 = self.branch2(x2)
        f3 = self.branch3(x3)
        fused = self.fusion(f1, f2, f3)
        return self.out(fused)


# ================================================================
# 2. Global Variables for Loading Models
# ================================================================
model_low = None
model_high = None
scalers_low = None
scalers_high = None


# ================================================================
# 3. Load Models & Scalers (must run once before prediction)
# ================================================================
def load_creep_models():

    global model_low, model_high, scalers_low, scalers_high

    # ---------------- Low-temperature model ----------------
    model_low = torch.load("model_low.pth", map_location=device)
    model_low.eval()

    scalers_low = joblib.load("scalers_low.pkl")
    # scalers_low = (sc1_low, sc2_low, sc3_low)

    # ---------------- High-temperature model ----------------
    model_high = torch.load("model_high.pth", map_location=device)
    model_high.eval()

    scalers_high = joblib.load("scalers_high.pkl")
    # scalers_high = (sc1_high, sc2_high, sc3_high)

    print("✅ Creep models & scalers loaded.")


# ================================================================
# 4. Prediction Wrapper for NSGA-II
# ================================================================
def predict_creep_life(feature_vector):
    """

        0-14:  15 chemical compositions
        15-16: stress, temperature(℃)
        17-21: Δδ, ΔS_mix, VEC, D_eff, T/Tm, σ/T

    Return:
        predicted_creep_life_hours (float)
    """

    if model_low is None:
        raise RuntimeError("❌ ERROR: Run load_creep_models() before prediction.")

    fv = np.array(feature_vector).reshape(1, -1)

    # ---- select model by T/Tm ----
    T_over_Tm = fv[0, 20]

    if T_over_Tm < 0.4:
        model = model_low
        sc1, sc2, sc3 = scalers_low
    else:
        model = model_high
        sc1, sc2, sc3 = scalers_high

    # ---- split into branches ----
    X2 = fv[:, 0:15]       # 15 composition
    X1 = fv[:, 15:17]      # stress, temperature
    X3 = fv[:, 17:22]      # physics features

    # ---- normalize ----
    X1n = sc1.transform(X1)
    X2n = sc2.transform(X2)
    X3n = sc3.transform(X3)

    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    X1t = to_t(X1n)
    X2t = to_t(X2n)
    X3t = to_t(X3n)

    # ---- predict ----
    with torch.no_grad():
        y_pred = model(X1t, X2t, X3t).item()

    # ---- convert log10 -> hours ----
    return float(10 ** y_pred)
