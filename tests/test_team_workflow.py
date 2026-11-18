"""Interface and alignment checks for the team benchmark."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from ensemble_team import combine

@pytest.mark.parametrize('name', ['ridge', 'lasso', 'random_forest', 'mlp'])
def test_model_interface(name):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 5))
    y = rng.normal(size=(40, 24))
    model = importlib.import_module(f'models.{name}.model').build_model()
    model.fit(x, y)
    result = model.predict(x[:3])
    assert result.shape == (3, 24)
    assert np.isfinite(result).all()

def test_ensemble_checks_alignment(tmp_path):
    frame = pd.DataFrame({'HourUTC':['2023-01-01T00:00:00Z', '2023-01-01T01:00:00Z'],
                          'day':['2023-01-01'] * 2, 'hour':[1,2],
                          'actual':[20.,30.], 'prediction':[10.,40.]})
    first, second = tmp_path / 'a.csv', tmp_path / 'b.csv'
    frame.to_csv(first, index=False)
    frame['prediction'] = [30.,20.]
    frame.to_csv(second, index=False)
    assert combine([first, second]).prediction.tolist() == [20.,30.]
    frame.loc[0, 'actual'] = 21.
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match='matching'):
        combine([first, second])
