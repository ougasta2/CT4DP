# Project proposal: Regime-aware day-ahead electricity price forecasting

**Course:** 02807 Computational Tools for Data Science, Fall 2026\
**Team size:** Five students\
**Status:** Proposal for TA feedback; model extensions and course-topic integration described below are planned, not completed.

## Motivation and research question

Day-ahead electricity prices vary with time, weather and changing market conditions. A single forecasting method may perform differently in different market situations. We propose to investigate whether grouping historical market days into interpretable regimes can help us combine multiple forecasting models for the DK1 electricity bidding zone in western Denmark.

Our main question is: **Can a clustering-based, regime-aware combination of price forecasts improve upon a simple equal-weight combination, and what are the trade-offs in forecast error and computational cost?** We will also compare the individual models and analyze performance by price range and time of day. An improvement is a question to test, not an assumed outcome or a required positive result.

## Prediction task and data

We aim to predict the next day's hourly mean DK1 day-ahead price in EUR/MWh using information intended to be available by 10:00 Copenhagen time on the preceding day. Historical inputs include lagged prices for DK1, DK2, Germany, the Netherlands and southern Sweden; calendar variables; and archived weather forecasts. The historical pipeline combines Danish Energinet price records with neighboring price series and archived weather data. After the 2025 market transition, quarter-hourly prices are averaged to an hourly target; the project does not claim quarter-hourly forecasts.

The timing of publication and historical revisions has not yet been verified for every input. We will document these gaps, audit sources where possible, and describe results as exploratory if point-in-time availability cannot be established. We will not use realized generation or same-auction prices as if they were available at the forecast cutoff.

## Proposed methods

### Course topic: general clustering

We propose to cluster **historical days** using standardized features that are intended to be available by the forecast cutoff, such as lagged price curves and weather forecasts. K-means will be the starting method; we will consider a second clustering method or a sensitivity analysis if feasible. We will investigate the number of clusters, cluster sizes, stability and interpretability, using suitable diagnostics such as the Davies–Bouldin index.

Clustering will be part of the forecasting method itself: for each forecast, the past-fitted cluster model assigns the day to a regime. The regime will be used to choose or weight the component forecasts based on **earlier out-of-fold errors**, not on the unknown target price. We will compare this regime-aware ensemble against an equal-weight ensemble on exactly the same observations. If the data are insufficient to estimate regime-specific weights reliably, we will use a simpler training-only rule and report the limitation. Feature scaling, clustering, and any weight selection will be fitted only on data preceding each evaluation period.

This integration is **proposed and not yet implemented or validated** in the new team benchmark. We would especially appreciate the TA's view on whether this use of clustering is sufficiently central and technically substantial for the course.

### Forecasting models and additional methods

Four initial model families have been set up as separate workspaces: Ridge regression, Lasso regression, Random Forest, and a feed-forward neural network (MLP). Each produces a 24-hour price curve using the same benchmark inputs. Each team member responsible for a model will investigate improvements and explain the choices and trade-offs. Lasso here is a regularized regression baseline inspired by the general LEAR literature, not an exact reproduction of the published LEAR benchmark.

The project lead plans to investigate the ensemble, starting with an equal-weight combination and then the clustering-based approach above. **Time-series block bootstrap** is a candidate method beyond the lectures for estimating uncertainty in paired differences between models while retaining short-range dependence. We plan to implement and explain it, and ask the TA to confirm whether it is an appropriate out-of-course method. Forecasting models and ensemble methods are additional techniques; we do not assume that their use alone satisfies a listed course-topic requirement.

A second course topic, frequent-itemset mining with Apriori, is available from an earlier exploratory study in the repository. We may use it to investigate interpretable patterns in forecast errors if it supports the main question and time permits; this is not yet integrated into the new benchmark, and we will not count it as a substantive course topic unless it becomes central and is validated. We welcome the TA's advice on whether to include it or concentrate on one substantial clustering topic.

## Evaluation

We plan monthly walk-forward evaluation: models are trained on earlier observations, then evaluated on the next month, with preprocessing fitted inside each training window. All methods and ensembles will be compared on the same delivery hours. We will include the previous-day same-hour price as a simple baseline, and compare the four individual models with the equal-weight and regime-aware ensembles.

The principal accuracy measure will be mean absolute error over all hours. We will also report RMSE, the proportion of predictions within 5% of the actual price for hours with |actual price| at least EUR 20/MWh, negative-price errors, and training/prediction time. Computation times will be measured under comparable hardware and settings. We will report denominators, missing data, model choices and unfavorable results.

The repository has previously been explored using 2023–2026 data, so these historical periods should not be described as a pristine final holdout. The current benchmark uses 2023–2024 for development. We would like the TA's advice on a realistic evaluation strategy, including whether a frozen prospective period or a clearly labeled historical comparison is sufficient given the course deadline.

## Current progress and remaining work

- **Already done:** data collection and processing for several earlier experiments; a shared model interface and monthly benchmark; initial Ridge, Lasso, Random Forest and MLP implementations; and a simple equal-weight ensemble with alignment checks. A one-month run of each model and the ensemble has been completed as an integration check. The Lasso and MLP runs produced convergence warnings that need investigation.
- **Still to do:** implement and validate the regime-aware method and any uncertainty analysis; improve and compare the models across agreed periods; audit source availability; analyze results; and prepare a final report and a documented, reproducible Jupyter notebook. Existing historical analyses and the new benchmark use different protocols and should not be presented as one completed experiment.

## Proposed division of work

The project lead has developed the initial data and experimental foundation and will coordinate the common benchmark, investigate ensemble methods and integrate the final report. Four team members will each take responsibility for one forecasting model: understanding and improving it, documenting experiments, analyzing results and drafting the relevant report section. The team will review the data audit, interpretation and final submission together. Contributions will be recorded explicitly, including work completed before this proposed division.

## Questions for the TA

1. Is regime-aware forecast combination a suitable use of general clustering as a central course topic? What level of implementation and analysis would be sufficiently substantial for a five-person team?
2. Would a block bootstrap be acceptable as the required method not directly taught in the lectures? Should we add Apriori as a second course topic, or focus on clustering and a more thorough evaluation?
3. Is the proposed division of work and existing foundation suitable for a five-person project? Are the benchmark, evaluation plan, data-vintage limitations and proposed deliverables appropriate?

## Course references

The [2026 course project description](https://courses.compute.dtu.dk/02807/2026/project/project.pdf) requires at least one, preferably two, course topics and at least one method not directly taught in lectures; it also asks for a technically challenging project, a documented notebook and individual contribution outline. The [2026 course schedule](https://courses.compute.dtu.dk/02807/2026/) lists general clustering and frequent itemsets among the taught topics. The proposal requests feedback rather than claiming that the requirements have already been satisfied.
