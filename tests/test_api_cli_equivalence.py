"""
Tests to verify Python API and CLI equivalence.

This module tests that the Python API behaves the same way as the CLI tool,
specifically for settings auto-detection and default behavior.
"""

import sys
import tempfile
import pytest
import json
import os
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

import neutron_event_analyzer as nea
from test_association_validation import (
    create_synthetic_photon_data,
    create_synthetic_event_data,
    create_synthetic_pixel_data,
    write_csv_files
)


@pytest.fixture
def complete_data_scenario():
    """Create a complete test scenario with photons and events."""
    event_configs = [
        {
            'event_id': 0,
            'center_x': 100.0,
            'center_y': 100.0,
            't_ns': 1000.0,
            'n_photons': 8,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        },
        {
            'event_id': 1,
            'center_x': 200.0,
            'center_y': 150.0,
            't_ns': 10000.0,
            'n_photons': 8,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        }
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    return {
        'photons': photon_df,
        'events': event_df,
        'configs': event_configs
    }


class TestAPICLIEquivalence:
    """Test that Python API behaves like the CLI."""

    def test_auto_detect_settings_on_init(self, complete_data_scenario):
        """Test that settings are auto-detected during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Create settings file
            settings = {
                'pixel2photon': {
                    'dSpace': 10.0,
                    'dTime': 1000e-9
                },
                'photon2event': {
                    'dSpace_px': 50.0,
                    'dTime_s': 500e-9
                }
            }

            settings_file = data_path / 'parameterSettings.json'
            with open(settings_file, 'w') as f:
                json.dump(settings, f)

            # Initialize without explicit settings
            analyser = nea.Analyse(data_folder=str(data_path))

            # Should have auto-detected the settings
            assert analyser.settings == settings
            assert 'parameterSettings.json' in analyser.settings_source

    def test_associate_uses_settings_defaults(self, complete_data_scenario):
        """Test that associate() uses settings file defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Write data files
            write_csv_files(
                None,  # No pixels
                complete_data_scenario['photons'],
                complete_data_scenario['events'],
                data_path,
                file_index=0
            )

            # Create settings file with specific parameters
            settings = {
                'pixel2photon': {
                    'dSpace': 8.0,
                    'dTime': 800e-9
                },
                'photon2event': {
                    'dSpace_px': 40.0,
                    'dTime_s': 400e-9
                }
            }

            settings_file = data_path / 'parameterSettings.json'
            with open(settings_file, 'w') as f:
                json.dump(settings, f)

            # Initialize and load
            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)

            # Get the defaults that will be used
            defaults = analyser._get_association_defaults()

            # Verify defaults match settings
            assert defaults['pixel_max_dist_px'] == 8.0
            assert defaults['pixel_max_time_ns'] == 800.0  # 800e-9 * 1e9
            assert defaults['photon_dSpace_px'] == 40.0
            assert defaults['max_time_ns'] == 400.0  # 400e-9 * 1e9

    def test_full_workflow_like_cli(self, complete_data_scenario):
        """Test full workflow: auto-detect settings, associate, save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Write data files
            write_csv_files(
                None,  # No pixels
                complete_data_scenario['photons'],
                complete_data_scenario['events'],
                data_path,
                file_index=0
            )

            # Create settings file
            settings = {
                'pixel2photon': {
                    'dSpace': 10.0,
                    'dTime': 1000e-9
                },
                'photon2event': {
                    'dSpace_px': 50.0,
                    'dTime_s': 500e-9
                }
            }

            settings_file = data_path / 'parameterSettings.json'
            with open(settings_file, 'w') as f:
                json.dump(settings, f)

            # Mimic CLI workflow
            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate(verbosity=0)  # Should use settings automatically

            # Verify association was performed
            assert analyser.associated_df is not None
            assert len(analyser.associated_df) > 0

            # Save results
            output_path = analyser.save_associations(verbosity=0)

            # Verify file was saved
            assert os.path.exists(output_path)

    def test_verbosity_shows_settings_source(self, complete_data_scenario):
        """Test that verbosity >= 1 shows settings source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Write minimal data
            write_csv_files(
                None,  # No pixels
                complete_data_scenario['photons'],
                complete_data_scenario['events'],
                data_path,
                file_index=0
            )

            # Create settings file
            settings = {
                'photon2event': {
                    'dSpace_px': 50.0,
                    'dTime_s': 500e-9
                }
            }

            settings_file = data_path / 'parameterSettings.json'
            with open(settings_file, 'w') as f:
                json.dump(settings, f)

            # Initialize
            analyser = nea.Analyse(data_folder=str(data_path))

            # Capture output when loading with verbosity
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                analyser.load(verbosity=1)

            output = f.getvalue()

            # Should mention settings source
            assert '⚙️' in output or 'settings' in output.lower()

    def test_explicit_settings_override_auto_detection(self):
        """Test that explicit settings override auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Create a settings file in data folder
            auto_settings = {
                'photon2event': {
                    'dSpace_px': 50.0,
                    'dTime_s': 500e-9
                }
            }

            settings_file = data_path / 'parameterSettings.json'
            with open(settings_file, 'w') as f:
                json.dump(auto_settings, f)

            # Use explicit preset instead
            analyser = nea.Analyse(data_folder=str(data_path), settings='in_focus')

            # Should use the explicit preset, not the auto-detected file
            assert analyser.settings_source == "preset 'in_focus'"
            # Settings should be from the preset, not the file
            from neutron_event_analyzer.config import DEFAULT_PARAMS
            assert analyser.settings == DEFAULT_PARAMS['in_focus']

    def test_hidden_settings_file_precedence(self):
        """Test that .parameterSettings.json takes precedence over parameterSettings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Create both visible and hidden settings files with different values
            visible_settings = {
                'photon2event': {'dSpace_px': 50.0, 'dTime_s': 500e-9}
            }
            hidden_settings = {
                'photon2event': {'dSpace_px': 60.0, 'dTime_s': 600e-9}
            }

            (data_path / 'parameterSettings.json').write_text(json.dumps(visible_settings))
            (data_path / '.parameterSettings.json').write_text(json.dumps(hidden_settings))

            # Initialize
            analyser = nea.Analyse(data_folder=str(data_path))

            # Should use hidden file
            assert analyser.settings == hidden_settings
            assert '.parameterSettings.json' in analyser.settings_source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
