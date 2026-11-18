# Historical DK1 demand and price model mathematics

The historical experiments predict next-day DK1 hourly mean prices. Most use a nominal 10:00 D-1 cutoff; EXAA experiments use a separate 11:00 cutoff. The primary historical hit metric is `abs(y - y_hat) <= 0.05 * abs(y)` for `abs(y) >= 20 EUR/MWh`; MAE uses all hours. Quarter-hourly prices after October 2025 are averaged to hourly targets.

## Demand

The alternative load model in [train_dk1_vanilla_load.py](../scripts/train_dk1_vanilla_load.py) uses calendar effects and temperature polynomials and interactions, based on [Ziel and Liu](https://doi.org/10.1016/j.ijforecast.2016.01.001):

```text
L_t = b0 + b_M[M_t] + b_W[W_t] + b_H[H_t] + b_WH[W_t,H_t]
    + sum(k=1..3) {a_k T_t^k + a_M[k,M_t] T_t^k + a_H[k,H_t] T_t^k} + error_t
```

Its small Ridge penalty, forecast temperatures, monthly refit and two-year window differ from the paper's probabilistic method. Another multi-output Ridge predicts 24-hour load curves from lagged consumption, weather and calendar features. Load forecasts used downstream are generated from earlier training data; realized future load is not used as a price predictor. See [demand research status](../results/dk1_demand_price/RESEARCH_STATUS.md) for detailed variants.

## Price

The LEAR-inspired model in [train_dk1_demand_price.py](../scripts/train_dk1_demand_price.py) uses regularized linear regression, following the general idea of [Lago et al.](https://doi.org/10.1016/j.apenergy.2021.116983), but with different inputs, fixed regularization and monthly refits:

```text
p_hat[d,h] = b_h + sum_j beta[h,j] X[d,j]
beta_hat_h = argmin_beta {(1/2n) sum_d (p[d,h] - b_h - beta_h dot X[d])^2
                           + lambda * ||beta_h||_1}
```

Inputs include lagged prices, out-of-fold load forecasts, forecast weather and lagged fuel proxies, with transformations fitted within training windows. [train_dk1_nonlinear.py](../scripts/train_dk1_nonlinear.py) also tests boosted trees on past out-of-fold price predictions or residuals. Source publication times and revisions remain incompletely audited; revised ENTSO-E load backfills are evaluated separately. The EXAA 11:00 horizon is distinct from 10:00 forecasts. No prior result establishes a 50% hit rate. The new team models are documented separately in [models/README.md](../models/README.md).
