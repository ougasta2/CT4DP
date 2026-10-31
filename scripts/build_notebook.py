"""Create the portable analysis companion using nbformat."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[
md('# Danish renewable regimes and electricity prices\n\n## tl;dr\nThis notebook reruns the cached official-data pilot. The executed findings below are the controlling empirical summary. No causal or operational forecasting interpretation is supported.'),
md('## Context & Methods\n2023 training and 2024 out-of-time validation, DK1/DK2. Essential course topics: K-means and independently implemented Apriori. Additional method: synchronous seven-day moving-block bootstrap.\n\n### Key Assumptions\nGross consumption is a settlement-derived demand proxy; solar self-consumption is estimated. Rules and all transformations are chosen on 2023 only. High price means above the fixed area-specific 2023 95th percentile. Units: MWh per hourly interval and EUR/MWh. See README for uncertainty, missing-value and model-selection details.'),
code("from pathlib import Path\nimport sys, subprocess\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'scripts/run_pilot.py').exists(), 'Open within this repository'\nsys.path.insert(0, str(ROOT / 'src'))\nRESULTS = ROOT / 'results'"),
md('## Data\nSource: Energinet Energi Data Service, CC BY 4.0. [Production metadata](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement), [price metadata](https://api.energidataservice.dk/meta/dataset/Elspotprices). Exact cached URL/parameter/download-time envelopes are in data/cache. Local year membership and DST are validated from UTC keys.\n\n### Reproduce from cached responses\nRun the live CLI once if the cache is unavailable. No network requests are made by this cell.'),
code("execution = subprocess.run([sys.executable, str(ROOT / 'scripts/run_pilot.py'), '--offline'], cwd=ROOT, capture_output=True, text=True)\nif execution.returncode:\n    raise RuntimeError(execution.stdout[-3000:] + execution.stderr[-3000:])\nprint('Cached pipeline completed successfully.')\ncoverage = pd.read_csv(RESULTS / 'coverage.csv')\ndisplay(coverage.groupby(['source', 'area'])[['expected','observed','missing','null_cells']].sum())"),
md('## Results\n### Event counts and baselines'),
code("display(pd.read_csv(RESULTS / 'event_rates.csv'))\ndisplay(pd.read_csv(RESULTS / 'k_diagnostics.csv'))\ndisplay(pd.read_csv(RESULTS / 'calendar_baseline.csv'))"),
md('### A preselected validation week\nThe first seven days are fixed before inspecting prices. UTC axes avoid repeated local DST hours.'),
code("display(Image(filename=str(RESULTS / 'figures/week.png')))"),
md('### Energy regimes\nCells require more than 20 observed hours; empty cells are not zeros.'),
code("display(Image(filename=str(RESULTS / 'figures/regimes.png')))"),
md('### Cluster interpretation\nCentroids use unscaled dimensionless physical ratios; cluster labels have no ordinal meaning.'),
code("display(pd.read_csv(RESULTS / 'cluster_centers.csv'))\ndisplay(Image(filename=str(RESULTS / 'figures/clusters.png')))"),
md('### Frozen rules and uncertainty\nLift uses the matching area and year baseline. Intervals are exploratory 95% seven-day moving-block bootstrap intervals and do not correct selection bias.'),
code("rules = pd.read_csv(RESULTS / 'selected_rules.csv')\ndisplay(rules[['event','antecedent','train_n','train_events','train_lift','test_n','test_events','test_lift','ci_low','ci_high','test_small_sample']])\nif (RESULTS / 'figures/rules.png').exists():\n    display(Image(filename=str(RESULTS / 'figures/rules.png')))"),
md('### Estimated solar self-consumption sensitivity\nThese scores keep the primary frozen centroids and rule definitions while changing the solar feature. The separate sensitivity folder also contains a full training-only refit.'),
code("display(pd.read_csv(RESULTS / 'sensitivity_comparison.csv'))"),
md('## Takeaways\nThe following report is generated from this execution, including whether the energy grouping improves over the calendar-only baseline. Inspect exact event denominators before interpreting negative-price enrichment.'),
code("display(Markdown((RESULTS / 'pilot_findings.md').read_text(encoding='utf-8')))"),
]
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}}
nbf.validate(nb)
(root/'notebooks').mkdir(exist_ok=True)
nbf.write(nb,root/'notebooks/energy_price_pilot.ipynb')
print('Notebook created and format validated.')
