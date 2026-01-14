# Creep–Fatigue Design of RAFM Steels Using Physics-Informed and Data-Driven Surrogate Models with Multi-Objective Optimisation
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
import torch, torch.nn as nn, torch.autograd as autograd, torch.utils.data as Data
import numpy as np, pandas as pd, matplotlib.pyplot as plt, optuna, warnings, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('TkAgg')

# ================================================================
# Basic settings
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = 8.314
print(f"Running on device: {device}")

# ================================================================
# Step 1. Data loading and regime separation
# ================================================================
FILE, TARGET = "creep_physics_features.xlsx", "Time to rupture"
df = pd.read_excel(FILE)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df[TARGET] > 0].reset_index(drop=True)

df_low  = df[df["T/Tm"] < 0.4].reset_index(drop=True)
df_high = df[df["T/Tm"] >= 0.4].reset_index(drop=True)

print(f"✅ Data loaded successfully: {len(df)} samples | Low-T={len(df_low)}, High-T={len(df_high)}")

# ================================================================
# Step 2. Model definitions
# ================================================================
class DeepBranch(nn.Module):
    def __init__(self, in_dim, hidden, depth, dropout):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden, hidden),
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
        self.proj1, self.proj2, self.proj3 = [nn.Linear(d, m) for d in dims]
        self.W = nn.Linear(m * 3, 3)
        self.proj_out = nn.Linear(m, m)

    def forward(self, f1, f2, f3):
        f1p, f2p, f3p = self.proj1(f1), self.proj2(f2), self.proj3(f3)
        scores = torch.softmax(self.W(torch.cat([f1p, f2p, f3p], dim=1)), dim=1)
        fused = scores[:, 0:1] * f1p + scores[:, 1:2] * f2p + scores[:, 2:3] * f3p
        return self.proj_out(fused)


class DeepHybridPINN(nn.Module):
    def __init__(self, dim1, dim2, dim3,
                 h1, h2, h3, d1, d2, d3, dr1, dr2, dr3, k_phys=0.5):
        super().__init__()
        self.b1 = DeepBranch(dim1, h1, d1, dr1)
        self.b2 = DeepBranch(dim2, h2, d2, dr2)
        self.b3 = DeepBranch(dim3, h3, d3, dr3)
        self.fuse = AttentionFusion([h1, h2, h3])
        self.out = nn.Sequential(
            nn.Linear(min(h1, h2, h3), 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Learnable physical parameters
        self.A = nn.Parameter(torch.tensor(10.0))
        self.k_phys = nn.Parameter(torch.tensor(k_phys))
        self.Q = nn.Parameter(torch.tensor(2.5e5))
        self.n = nn.Parameter(torch.tensor(5.0))

    def forward(self, x1, x2, x3):
        f1, f2, f3 = self.b1(x1), self.b2(x2), self.b3(x3)
        return self.out(self.fuse(f1, f2, f3))

    def physics_prior(self, x1, sc1):
        stress, T = x1[:, 0:1], x1[:, 1:2]
        T = T * sc1.scale_[1] + sc1.mean_[1]
        stress = stress * sc1.scale_[0] + sc1.mean_[0]

        T = torch.clamp(T + 273.15, 200.0)
        stress = torch.clamp(stress, 1.0)

        n_eff = torch.clamp(self.n, 1.0, 10.0)
        Q_eff = torch.clamp(self.Q, 1.5e5, 4.0e5)

        y_phys = self.A - n_eff * torch.log(stress) + Q_eff / (R * T)
        return y_phys * self.k_phys

# ================================================================
# Step 3. Data preparation
# ================================================================
def prepare_data(df):
    b1 = ["stress", "temperature"]
    b2 = df.columns[:15]
    b3 = ["Δδ", "ΔS_mix", "VEC", "D_eff", "T/Tm", "σ/T"]

    X1, X2, X3 = df[b1].values, df[b2].values, df[b3].values
    y = np.log10(df[TARGET].values + 1).reshape(-1, 1)

    sc1, sc2, sc3 = StandardScaler(), StandardScaler(), StandardScaler()
    X1 = sc1.fit_transform(X1)
    X2 = sc2.fit_transform(X2)
    X3 = sc3.fit_transform(X3)

    X1_tr, X1_te, X2_tr, X2_te, X3_tr, X3_te, y_tr, y_te = train_test_split(
        X1, X2, X3, y, test_size=0.2, random_state=42
    )

    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    X1_tr, X2_tr, X3_tr, y_tr = map(to_t, [X1_tr, X2_tr, X3_tr, y_tr])
    X1_te, X2_te, X3_te, y_te = map(to_t, [X1_te, X2_te, X3_te, y_te])

    loader = Data.DataLoader(
        Data.TensorDataset(X1_tr, X2_tr, X3_tr, y_tr),
        batch_size=64,
        shuffle=True
    )
    return X1_tr, X2_tr, X3_tr, y_tr, X1_te, X2_te, X3_te, y_te, loader, sc1

# ================================================================
# Step 4. Hybrid loss function with physics constraints
# ================================================================
def hybrid_loss(model, x1, x2, x3, y_true, λ1, λ2, λ3, λ4, sc1):
    y_pred = model(x1, x2, x3)
    y_phys = model.physics_prior(x1, sc1)

    L_data = torch.mean((y_pred - y_true) ** 2)
    L_soft = torch.mean((y_pred - y_phys) ** 2)

    stress = x1[:, 0:1].clone().detach().requires_grad_(True)
    temp = x1[:, 1:2].clone().detach().requires_grad_(True)
    y_temp = model(torch.cat([stress, temp], dim=1), x2, x3)

    grad_T = autograd.grad(y_temp.sum(), temp, create_graph=True, allow_unused=True)[0]
    grad_sigma = autograd.grad(y_temp.sum(), stress, create_graph=True, allow_unused=True)[0]

    L_mono = torch.relu(grad_T).mean() + torch.relu(grad_sigma).mean()
    L_bound = torch.mean(torch.exp(y_pred[y_pred > y_pred.mean()])) \
        if (y_pred > y_pred.mean()).any() else torch.tensor(0., device=device)
    L_grad = torch.mean(torch.abs(grad_T)) + torch.mean(torch.abs(grad_sigma))

    L_total = L_data + λ1 * L_soft + λ2 * L_mono + λ3 * L_bound + λ4 * L_grad
    return L_total, L_data.item(), L_soft.item(), float(L_mono), float(L_bound), float(L_grad)

# ================================================================
# Step 5. Optuna objective function
# ================================================================
def objective(trial, df, tag):
    X1_tr, X2_tr, X3_tr, y_tr, X1_te, X2_te, X3_te, y_te, loader, sc1 = prepare_data(df)

    h1 = trial.suggest_categorical("h1", [64, 96, 128, 160])
    h2 = trial.suggest_categorical("h2", [64, 96, 128, 160])
    h3 = trial.suggest_categorical("h3", [64, 96, 128, 160])
    d1 = trial.suggest_int("d1", 1, 4)
    d2 = trial.suggest_int("d2", 1, 4)
    d3 = trial.suggest_int("d3", 1, 4)
    dr1 = trial.suggest_float("dr1", 0.1, 0.5)
    dr2 = trial.suggest_float("dr2", 0.1, 0.5)
    dr3 = trial.suggest_float("dr3", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    λ1 = trial.suggest_float("λ1", 1e-5, 1e-2, log=True)
    λ2 = trial.suggest_float("λ2", 1e-5, 1e-2, log=True)
    λ3 = trial.suggest_float("λ3", 1e-5, 1e-2, log=True)
    λ4 = trial.suggest_float("λ4", 1e-5, 1e-2, log=True)

    model = DeepHybridPINN(
        X1_tr.shape[1], X2_tr.shape[1], X3_tr.shape[1],
        h1, h2, h3, d1, d2, d3, dr1, dr2, dr3
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(200):
        for xb1, xb2, xb3, yb in loader:
            opt.zero_grad()
            L, *_ = hybrid_loss(model, xb1, xb2, xb3, yb, λ1, λ2, λ3, λ4, sc1)
            L.backward()
            opt.step()

    with torch.no_grad():
        yp = model(X1_te, X2_te, X3_te)

    y_true_exp = 10 ** y_te.cpu().numpy()
    y_pred_exp = 10 ** yp.cpu().numpy()
    r2 = r2_score(y_true_exp, y_pred_exp)

    trial.set_user_attr("R2", r2)
    return -r2
