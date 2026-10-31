# Implementation notes

- Repository initially contained only the task Markdown; no existing code to reuse.
- Official metadata and API guide verified before full acquisition. The API's
  `start`/`end` dates use Copenhagen local time, while `HourUTC` is UTC without Z.
- Initial Python sandbox network attempt failed DNS. The authorized network-enabled
  attempt succeeded. All empirical inputs come from the original public API.
- Initial price smoke request returned HTTP 429 with Retry-After=108. The client
  now honors bounded waits. Annual rather than monthly download partitions reduce
  request pressure; coverage remains audited monthly. No repeated polling of
  unchanged historical responses is needed because successful responses are cached.
- Four-day/three-day smoke split is an execution diagnostic only. It is isolated
  in smoke_only and explicitly labeled; it is not used in final findings.
- Default thresholds: antecedent support 2%, at most 3 items, >=100 antecedent
  hours and >=20 events for training selection; max one rule per area/outcome.
  Small-sample validation flags use the same minimum counts. These defaults were
  fixed before inspecting validation outcomes.
- Empirical completion and observed test results are recorded separately in
  results/validation.md after execution. Source warnings about ambiguous zeros
  cannot be resolved by automatic cleaning and remain an explicit limitation.
