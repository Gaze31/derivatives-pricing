# Derivatives Pricing & Options Analytics

A Python library implementing core derivatives pricing models from scratch — covering closed-form solutions, numerical methods, and Greeks calculation. Built as part of a quantitative finance study to understand how theoretical pricing models behave under different market conditions.

---

## Models Implemented

| Model | File | Method |
|---|---|---|
| Black-Scholes | `black_scholes.py` | Closed-form analytical solution |
| Binomial Tree | `binomial_tree.py` | CRR lattice model |
| Monte Carlo Pricing | `monte_carlopricing.py` | GBM simulation |
| Asian Option Pricing | `asian_optionpricing.py` | Path-dependent Monte Carlo |
| Option Greeks | `greek_python.py` | Delta, Gamma, Theta, Vega, Rho |
| PDE Simulation | `pde_simulation.py` | Finite difference method |
| Brownian Motion | `browian_motionpy.py` | GBM path simulation |

---

## Sample Output

### Black-Scholes Call Price vs Strike
![BS Call vs Strike](/Users/sumedhahundekar/python/quant_eng/bs_call_vs_strike.png)

Shows the characteristic convex decay of call option value as strike price increases, with time value embedded at each point.

### Option Calibration
![Option Calibration](/Users/sumedhahundekar/python/quant_eng/exercise2_option_calibration.png)
---

## Key Concepts Covered

**Black-Scholes Model**
- European call and put pricing
- Implied volatility calculation
- Put-call parity verification

**Binomial Tree (CRR)**
- American and European option pricing
- Early exercise premium calculation
- Convergence to Black-Scholes as steps increase

**Monte Carlo Methods**
- Geometric Brownian Motion path generation
- Asian option pricing (arithmetic and geometric average)
- Confidence interval estimation

**Greeks**
- Delta: sensitivity to underlying price
- Gamma: rate of change of delta
- Theta: time decay
- Vega: sensitivity to volatility
- Rho: sensitivity to interest rates

---

## Setup

```bash
git clone https://github.com/Gaze31/derivatives-pricing.git
cd derivatives-pricing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python black_scholes.py
```

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
```

---

## Known Limitations

- Black-Scholes assumes constant volatility — does not account for volatility smile
- Monte Carlo pricing accuracy depends on number of simulations (computational tradeoff)
- No exotic options beyond Asian (no barrier, lookback, or digital)
- Models use risk-free rate as static input — no term structure

---

## Next Steps

- [ ] Implement volatility smile and surface modeling
- [ ] Add barrier option pricing
- [ ] Heston stochastic volatility model
- [ ] Interest rate models (Vasicek, CIR)

---

## Author

**Sumedha Hundekar** — Finance graduate building quantitative finance tools in Python.  
Contact: velvetgazeze@gmail.com
