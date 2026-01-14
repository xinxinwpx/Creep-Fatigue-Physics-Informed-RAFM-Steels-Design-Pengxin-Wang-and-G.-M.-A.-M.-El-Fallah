Creep–fatigue design of RAFM steels using physics-informed surrogate models and multi-objective optimisation.

## Authors
Pengxin Wang, G. M. A. M. El-Fallah

---

## Overview

This repository presents a physics-informed and data-driven modelling framework for the creep–fatigue design of reduced-activation ferritic/martensitic (RAFM) steels.

The framework integrates:
- Physics-informed surrogate modelling for creep life prediction
- Data-driven machine learning models for fatigue life prediction
- Modular encapsulation of creep and fatigue models
- NSGA-II–based multi-objective optimisation to balance creep and fatigue performance

The codebase is intended for reproducible research, scientific publication, and materials design studies.

---

## Contents

- Creep model encapsulation  
  Physics-informed surrogate model for creep life prediction.

- Fatigue model encapsulation  
  Data-driven fatigue life prediction using machine learning.

- NSGA-II–based multi-objective optimisation  
  Multi-objective optimisation of creep–fatigue performance.

- Creep dataset  
  Dataset for creep model training and validation.

- Fatigue dataset  
  Dataset for fatigue model training and validation.

---

## Repository Structure

.
├── README.md
├── creep_model.py
├── fatigue_model.py
├── Encapsulation of the creep model.py
├── Encapsulation of the fatigue model.py
├── NSGA-II-based multi-objective optimisation.py
├── creep_dataset.xlsx
└── fatigue_data.xlsx

---

## Usage

```bash
# 1. Clone the repository
git clone https://github.com/xinxinwpx/Creep-Fatigue-Physics-Informed-RAFM-Steels-Design-Pengxin-Wang-and-G.-M.-A.-M.-El-Fallah.git
cd Creep-Fatigue-Physics-Informed-RAFM-Steels-Design-Pengxin-Wang-and-G.-M.-A.-M.-El-Fallah

# 2. Install dependencies
pip install numpy pandas scikit-learn xgboost optuna matplotlib seaborn

# 3. Run the creep model
python creep_model.py
# or using the encapsulated version
python "Encapsulation of the creep model.py"

# 4. Run the fatigue model
python fatigue_model.py
# or using the encapsulated version
python "Encapsulation of the fatigue model.py"

# 5. Run NSGA-II-based multi-objective optimisation
python "NSGA-II-based multi-objective optimisation.py"
```

---

## Contact

For questions or collaboration, please contact:

Dr. Gebril El-Fallah
📧 gmae2@leicester.ac.uk

---

## License

This project is licensed under the MIT License.  
See the LICENSE file for details.

