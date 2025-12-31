"""
Example: Using settings file for association parameters

This script demonstrates how to use a settings JSON file or dictionary
to provide default association parameters based on empir configuration.
"""

import neutron_event_analyzer as nea

# Option 1: Load settings from JSON file
print("="*70)
print("Example 1: Using settings from JSON file")
print("="*70)

analyser = nea.Analyse(
    data_folder='../tests/data/neutrons',
    settings='settings_example.json'  # Path to settings JSON file
)

analyser.load(load_pixels=True, load_photons=True, load_events=True, verbosity=1)

# Association will use parameters from settings file
result_df = analyser.associate_full(verbosity=1)

print(f"\nAssociation complete: {len(result_df)} rows")
print(f"Pixels → Photons: {result_df['assoc_photon_id'].notna().sum()}")
print(f"Pixels → Events: {result_df['assoc_event_id'].notna().sum()}\n")


# Option 2: Use settings dictionary directly
print("="*70)
print("Example 2: Using settings dictionary")
print("="*70)

settings_dict = {
    "pixel2photon": {
        "dSpace": 10.0,           # 10 pixels
        "dTime": 1000e-09,        # 1000 ns
    },
    "photon2event": {
        "dSpace_px": 50.0,        # 50 pixels
        "dTime_s": 500e-09,       # 500 ns
    }
}

analyser2 = nea.Analyse(
    data_folder='../tests/data/neutrons',
    settings=settings_dict  # Pass dictionary directly
)

analyser2.load(load_pixels=True, load_photons=True, load_events=True, verbosity=1)

# Association will use parameters from settings dictionary
result_df2 = analyser2.associate_full(verbosity=1)

print(f"\nAssociation complete: {len(result_df2)} rows")
print(f"Pixels → Photons: {result_df2['assoc_photon_id'].notna().sum()}")
print(f"Pixels → Events: {result_df2['assoc_event_id'].notna().sum()}\n")


# Option 3: Override settings with explicit parameters
print("="*70)
print("Example 3: Override settings with explicit parameters")
print("="*70)

analyser3 = nea.Analyse(
    data_folder='../tests/data/neutrons',
    settings=settings_dict
)

analyser3.load(load_pixels=True, load_photons=True, load_events=True, verbosity=1)

# Explicitly provided parameters override settings
result_df3 = analyser3.associate_full(
    pixel_max_dist_px=15.0,  # Override the 10.0 from settings
    verbosity=1
)

print(f"\nAssociation complete: {len(result_df3)} rows")
print(f"Pixels → Photons: {result_df3['assoc_photon_id'].notna().sum()}")
print(f"Pixels → Events: {result_df3['assoc_event_id'].notna().sum()}\n")


print("="*70)
print("Settings feature demonstration complete!")
print("="*70)
