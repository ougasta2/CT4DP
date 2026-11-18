# Project status as of 10 November 2026

The earlier 10:00 D-1 DK1 study predicts hourly average prices. In its monthly rolling 2023/2024 comparison, previous-day same-hour MAE was 29.98/28.90 EUR/MWh, Ridge plus weather 18.78/18.34, and a LEAR-inspired raw Lasso with past-error correction 18.16/18.01. Their respective ±5% hit rates on hours with |actual| >= 20 were 16.93%/14.92%, 20.35%/17.80%, and 22.20%/18.16%. The correction was selected using 2023; both years informed development. See [full results and denominators](../results/dk1_rolling_benchmark/findings.md).

Archived weather, lagged fuel proxies and prices have not had every publication and revision time verified. The Energinet day-ahead renewable forecast is published at 18:00 D-1, too late for the 10:00 experiment. The earlier latest-year comparison uses a different period and protocol and is not directly comparable. See [research status](../results/dk1_demand_price/RESEARCH_STATUS.md) for subsequent demand, nonlinear and 11:00 EXAA experiments and their limitations.

This document is a historical snapshot, not the specification of the [new team benchmark](TEAM_WORKFLOW.md). Earlier local model files may not be in Git; an operational service and a faithful point-in-time source audit remain unfinished.
