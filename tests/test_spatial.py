import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('spatial', Path(__file__).parents[1] / 'scripts/run_spatial_model.py')
spatial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spatial)


def test_paired_states_and_difference_sign():
    rows = []
    for hour, prices in enumerate([(1, 2), (-1, -2), (-1, 2), (1, -2)]):
        for area, price, wind in [('DK1', prices[0], 2.), ('DK2', prices[1], 1.)]:
            rows.append(dict(HourUTC=f'2024-01-01T0{hour}:00Z', PriceArea=area,
                             SpotPriceEUR=price, wind_to_load=wind, solar_to_load=.2, load_index=1.))
    result = spatial.paired(pd.DataFrame(rows))
    assert result.state.tolist() == ['neither', 'both', 'DK1_only', 'DK2_only']
    assert result.discordant.tolist() == [False, False, True, True]
    assert result.diff_wind_to_load.eq(1.).all()
    with pytest.raises(ValueError, match='complete'):
        spatial.paired(pd.DataFrame(rows[:-1]))


def test_block_interval_preserves_constant_difference():
    frame = pd.DataFrame({'year':[2024,2024], 'local_date':pd.to_datetime(['2024-01-01','2024-06-01']).date})
    assert np.allclose(spatial.block_delta(frame, np.ones(2), repeats=100), [1,1])
