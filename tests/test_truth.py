"""Tests of the simulation-truth join (neutron_event_analyzer.truth)."""
import pandas as pd

from neutron_event_analyzer.truth import TICK_S, join_truth


def test_join_on_pixel_and_tick(tmp_path):
    (tmp_path / 'TracedPhotons').mkdir(); (tmp_path / 'SimPhotons').mkdir()
    pd.DataFrame({'pixel_x': [10, 11, 50], 'pixel_y': [20, 20, 60], 'toa2': [0, 0, 0], 'toa_tick': [100, 101, 5000],
                  'photon_count': 1, 'time_diff': 0, 'id': [1, 1, 2], 'sim_id': [1, 1, 2], 'neutron_id': [7, 7, 8],
                  'pulse_id': [3, 3, 4], 'pulse_time_ns': 0.0, 'in_tpx3': [True, True, False],
                  'x_opt': [10.4, 10.4, 50.0], 'y_opt': [20.1, 20.1, 60.0]}).to_csv(tmp_path / 'TracedPhotons' / 'traced_sim_data_0.csv', index=False)
    pd.DataFrame({'id': [1, 2], 'pulse_id': [3, 4], 'neutron_id': [7, 8], 'px': [1.0, 2.0], 'py': [3.0, 4.0], 'pz': [5.0, 6.0],
                  'parentName': ['proton', 'e-'], 'neutronEnergy': [5.0, 2.0], 'nx0': [1.1, 2.2], 'ny0': [3.3, 4.4]}
                 ).to_csv(tmp_path / 'SimPhotons' / 'sim_data_0.csv', index=False)
    assoc = pd.DataFrame({'px/x': [10.0, 11.0, 30.0], 'px/y': [20.0, 20.0, 30.0], 'px/toa': [100 * TICK_S, 101 * TICK_S, 7 * TICK_S],
                          'ph/id': [1, 1, 2], 'ev/id': [1, 1, 2]})
    j = join_truth(assoc, tmp_path)
    assert list(j['sim/neutron_id'][:2]) == [7, 7] and pd.isna(j['sim/neutron_id'][2])
    assert j['sim/parentName'][0] == 'proton' and j['sim/nx0'][1] == 1.1
    assert abs(j.attrs['truth_match'] - 2 / 3) < 1e-9
