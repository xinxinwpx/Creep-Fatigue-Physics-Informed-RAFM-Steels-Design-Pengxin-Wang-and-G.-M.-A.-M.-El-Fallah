# Creep–Fatigue Design of RAFM Steels Using Physics-Informed and Data-Driven Surrogate Models with Multi-Objective Optimisation
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
# ================================================================
import numpy as np
import random
import torch
import joblib
from tqdm import tqdm

# ================================================================
# A. LOAD MODELS (Creep + Fatigue)
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Fatigue ----------------
def load_fatigue_model():
    global fatigue_model, scaler_fatigue
    fatigue_model = joblib.load("fatigue_model.pkl")
    scaler_fatigue = joblib.load("scaler_fatigue.pkl")
    print("📌 Fatigue model loaded.")


def predict_fatigue_life(x):
    x = np.array(x).reshape(1, -1)
    x_norm = scaler_fatigue.transform(x)
    y_pred = fatigue_model.predict(x_norm)[0]
    return float(y_pred)


# ---------------- Creep ----------------
def load_creep_models():
    global model_low, model_high, scalers_low, scalers_high
    model_low = torch.load("model_low.pth", map_location=device)
    model_low.eval()

    model_high = torch.load("model_high.pth", map_location=device)
    model_high.eval()

    scalers_low = joblib.load("scalers_low.pkl")
    scalers_high = joblib.load("scalers_high.pkl")

    print("📌 Creep models loaded (Low-T & High-T).")


def predict_creep_life(stress, T_C, comp, phys):
    """
    comp = 15
    phys = 6 physics variables
    """
    T_K = T_C + 273.15

    X1 = np.array([stress, T_C]).reshape(1, -1)
    X2 = np.array(comp).reshape(1, -1)
    X3 = np.array(phys).reshape(1, -1)


    T_over_Tm = phys[4]

    if T_over_Tm < 0.4:
        model = model_low
        sc1, sc2, sc3 = scalers_low
    else:
        model = model_high
        sc1, sc2, sc3 = scalers_high


    X1n = sc1.transform(X1)
    X2n = sc2.transform(X2)
    X3n = sc3.transform(X3)

    X1t = torch.tensor(X1n, dtype=torch.float32, device=device)
    X2t = torch.tensor(X2n, dtype=torch.float32, device=device)
    X3t = torch.tensor(X3n, dtype=torch.float32, device=device)

    with torch.no_grad():
        y = model(X1t, X2t, X3t).item()

    return float(10 ** y)


# ================================================================
# B. Physics Feature Calculation (Creep)
# ================================================================
R = 8.314

ELEMENTS = ["C","Si","Mn","P","S","Ni","Mo","Cr","W","V","Co","Ta","Nb","Cu","Al"]

element_data = {
    'C':  {'M':12.01,'r':70,'Tm':4100,'VEC':4,'D0':1e-4,'Q':250e3},
    'Si': {'M':28.09,'r':111,'Tm':1687,'VEC':4,'D0':2e-5,'Q':250e3},
    'Mn': {'M':54.94,'r':127,'Tm':1519,'VEC':7,'D0':2e-5,'Q':245e3},
    'P':  {'M':30.97,'r':110,'Tm':317,'VEC':5,'D0':2e-5,'Q':250e3},
    'S':  {'M':32.06,'r':102,'Tm':388,'VEC':6,'D0':2e-5,'Q':250e3},
    'Ni': {'M':58.69,'r':124,'Tm':1728,'VEC':10,'D0':1.8e-5,'Q':260e3},
    'Mo': {'M':95.95,'r':136,'Tm':2896,'VEC':6,'D0':2e-5,'Q':280e3},
    'Cr': {'M':52.00,'r':128,'Tm':2180,'VEC':6,'D0':2.4e-5,'Q':250e3},
    'W':  {'M':183.84,'r':137,'Tm':3695,'VEC':6,'D0':1.5e-5,'Q':300e3},
    'V':  {'M':50.94,'r':134,'Tm':2183,'VEC':5,'D0':3.0e-5,'Q':240e3},
    'Co': {'M':58.93,'r':125,'Tm':1768,'VEC':9,'D0':2e-5,'Q':255e3},
    'Ta': {'M':180.95,'r':143,'Tm':3290,'VEC':5,'D0':1.5e-5,'Q':310e3},
    'Nb': {'M':92.91,'r':146,'Tm':2750,'VEC':5,'D0':2e-5,'Q':280e3},
    'Cu': {'M':63.55,'r':128,'Tm':1358,'VEC':11,'D0':2e-5,'Q':240e3},
    'Al': {'M':26.98,'r':143,'Tm':933,'VEC':3,'D0':2e-5,'Q':250e3}
}

def compute_physics(comp):
    moles = [comp[i] / element_data[ELEMENTS[i]]["M"] for i in range(15)]
    total = sum(moles)
    c = [m/total for m in moles]

    r_avg = sum(c[i] * element_data[ELEMENTS[i]]["r"] for i in range(15))

    delta = np.sqrt(sum(c[i] * (1 - element_data[ELEMENTS[i]]["r"]/r_avg)**2 for i in range(15)))
    deltaS = -R * sum(c[i]*np.log(c[i] + 1e-12) for i in range(15))
    VEC = sum(c[i] * element_data[ELEMENTS[i]]["VEC"] for i in range(15))

    Tm_mix = sum(c[i] * element_data[ELEMENTS[i]]["Tm"] for i in range(15))
    T_over_Tm = (650+273.15) / Tm_mix

    D_eff = sum(c[i] * element_data[ELEMENTS[i]]["D0"] *
                np.exp(-element_data[ELEMENTS[i]]["Q"]/(R*(650+273.15)))
                for i in range(15))

    sigma_T = 100 / (650+273.15)

    return [delta, deltaS, VEC, D_eff, T_over_Tm, sigma_T]


# ================================================================
# C. Fatigue input construction
# ================================================================
FIX_FATIGUE = {
    "Δε": 0.8,
    "σ1": 440,
    "σmax": 450,
    "soft": -0.15,
    "rate": 3e-3,
    "T": 823
}

def make_fatigue_input(comp):
    return list(comp) + [
        FIX_FATIGUE["Δε"],
        FIX_FATIGUE["σ1"],
        FIX_FATIGUE["σmax"],
        FIX_FATIGUE["soft"],
        FIX_FATIGUE["rate"],
        FIX_FATIGUE["T"]
    ]


# ================================================================
# D. Constraints (convert to feasible)
# ================================================================
MAX_SUM = 20.0
MAX_VAL = np.array([1,2,2,0.05,0.02,2,1,10,2,0.3,0.3,0.1,0.1,0.5,2])

def repair(comp):
    comp = np.clip(comp, 0, MAX_VAL)
    s = comp.sum()
    if s > MAX_SUM:
        comp = comp * (MAX_SUM / s)
    return comp


# ================================================================
# E. Multi-objective evaluation
# ================================================================
def evaluate(comp):
    comp = repair(comp)

    phys = compute_physics(comp)

    creep100 = predict_creep_life(100, 650, comp, phys)
    creep75  = predict_creep_life(75,  650, comp, phys)

    fatigue = predict_fatigue_life(make_fatigue_input(comp))

    return [-creep100, -creep75, -fatigue]


# ================================================================
# F. NSGA-II
# ================================================================
def nsga2(pop_size=60, gen=80):
    pop = [repair(np.random.rand(15)*MAX_VAL) for _ in range(pop_size)]

    for g in range(gen):
        print(f"Gen {g+1}/{gen}")

        new_pop = []
        for _ in range(pop_size):
            p1, p2 = random.sample(pop, 2)
            beta = random.random()
            child = beta*p1 + (1-beta)*p2
            if random.random()<0.3:
                idx = random.randint(0,14)
                child[idx] += np.random.randn()*0.1
            child = repair(child)
            new_pop.append(child)

        pop = new_pop

    return pop


# ================================================================
# G. MAIN
# ================================================================
if __name__ == "__main__":
    load_creep_models()
    load_fatigue_model()

    print("🚀 Running NSGA-II optimisation...")
    pop = nsga2()

    results = [(ind, evaluate(ind)) for ind in pop]
    results.sort(key=lambda x: sum(x[1]))

    print("\n🏆 Best 10 alloys:")
    for comp, f in results[:10]:
        print("\nComp:", comp)
        print("Creep100:", -f[0], " Creep75:", -f[1], " Fatigue:", -f[2])
