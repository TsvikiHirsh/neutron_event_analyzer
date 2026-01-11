# EMPIR Parameter Optimization Framework

## Overview

The EMPIR Parameter Optimization Framework provides automated parameter tuning for EMPIR reconstruction **without requiring ground truth or event association**. Instead of comparing to known answers, it analyzes intrinsic distribution shapes from each reconstruction stage to identify optimal parameters based on statistical signatures.

## Core Philosophy

Each EMPIR parameter controls a specific aspect of clustering (temporal, spatial, size limits) that leaves a distinctive statistical signature in the output distributions. By analyzing these signatures, we can:

1. **Identify over-clustering**: Parameters too loose, merging unrelated signals
2. **Identify under-clustering**: Parameters too tight, fragmenting true signals
3. **Find optimal values**: Where signal/noise separation is maximized

This approach works because good parameters produce clean, well-separated distributions, while poor parameters create artifacts, outliers, and mixed populations.

## Two-Stage Optimization

### Stage 1: Pixel-to-Photon Clustering

**Optimizes**: `dSpace`, `dTime`, `nPxMin`, `nPxMax`

These parameters control how individual detector pixels are grouped into photon clusters.

| Parameter | Description | Optimization Approach |
|-----------|-------------|----------------------|
| `dTime` | Temporal clustering window | Analyze intra-photon pixel time differences. Fit Gaussian to signal peak, set threshold at 3σ to capture signal while excluding noise. |
| `dSpace` | Spatial clustering radius | Analyze pixel distances from photon centroids. Find P95/P99 percentiles, adjust to maintain 1-3% outlier fraction. |
| `nPxMin` | Minimum pixels per photon | Analyze cluster size distribution. Set based on small-cluster fraction to reject noise without cutting signal. |
| `nPxMax` | Maximum pixels per photon | Analyze cluster size distribution. Set at P95-P99 to reject pile-up while keeping 98%+ of events. |

### Stage 2: Photon-to-Event Clustering

**Optimizes**: `dSpace_px`, `dTime_s`, `durationMax_s`

These parameters control how photons are grouped into neutron events.

| Parameter | Description | Optimization Approach |
|-----------|-------------|----------------------|
| `dTime_s` | Temporal clustering window | Analyze inter-photon time intervals. Fit two-component exponential (signal + background), set at crossover point for ~70% purity. |
| `dSpace_px` | Spatial clustering radius | Analyze photon distances from event centroids and event multiplicities. Adjust based on single-photon fraction and mean multiplicity. |
| `durationMax_s` | Maximum event duration | Analyze event duration distribution. Fit exponential, set at 5τ or P99 to keep 98%+ of events. |

## Quick Start

### CLI Usage (Recommended)

The simplest way to optimize EMPIR parameters is using the `nea-empir` command:

```bash
# Optimize photon-to-event parameters (most common use case)
nea-empir /path/to/data --stage photon2event --output optimized_params.json

# Optimize pixel-to-photon parameters
nea-empir /path/to/data --stage pixel2photon --output optimized_params.json

# Optimize both stages
nea-empir /path/to/data --stage both --output optimized_params.json

# Use current parameters as starting point for comparison
nea-empir /path/to/data --current-params current_settings.json --output new_params.json
```

**Output**: A JSON file in EMPIR `parameterSettings.json` format, ready to use for reconstruction.

### Python API Usage

For programmatic access or integration into analysis pipelines:

```python
import neutron_event_analyzer as nea

# Simple one-liner
suggestions = nea.optimize_empir_parameters(
    data_folder='/path/to/data',
    stage='both',  # or 'pixel2photon', 'photon2event'
    output_path='optimized_params.json',
    verbosity=1
)

# Access individual suggestions
print(suggestions['photon2event'])
print(suggestions['pixel2photon'])

# Get parameter JSON for EMPIR
params = suggestions['photon2event'].get_parameter_json()
```

### Advanced Usage: Fine-Grained Control

For more control over the optimization process:

```python
import neutron_event_analyzer as nea

# Load your data
analyser = nea.Analyse(data_folder='/path/to/data')
analyser.load(load_pixels=True, load_photons=True, load_events=True)

# Associate data (needed for most optimizations)
analyser.associate_photons_events(method='simple')

# Create optimizer
optimizer = nea.EMPIRParameterOptimizer(analyser, verbosity=1)

# Optimize pixel-to-photon stage
p2p_suggestion = optimizer.optimize_pixel2photon(
    current_dSpace=2.0,      # mm
    current_dTime=100e-9,    # seconds
    current_nPxMin=8,
    current_nPxMax=100
)

print(p2p_suggestion)
print(f"Suggested dTime: {p2p_suggestion.suggested_params['dTime']*1e9:.1f} ns")
print(f"Reasoning: {p2p_suggestion.reasoning['dTime']}")

# Optimize photon-to-event stage
p2e_suggestion = optimizer.optimize_photon2event(
    current_dSpace_px=50.0,
    current_dTime_s=50e-9,
    current_durationMax_s=500e-9
)

print(p2e_suggestion)

# Save to file
import json
with open('optimized_params.json', 'w') as f:
    json.dump(p2e_suggestion.get_parameter_json(), f, indent=2)
```

## Understanding the Output

### Console Output

```
======================================================================
EMPIR Parameter Suggestions
======================================================================

PIXEL2PHOTON:
----------------------------------------------------------------------
  dSpace          2.0          → 2.35         ↑ (+17.5%) [high]
    • Outlier fraction 6.2% > 5% (too tight)
    • Setting dSpace = P99 = 2.35 px

  dTime           1e-07        → 8.5e-08      ↓ (-15.0%) [high]
    • Fitted Gaussian: μ=25.3 ns, σ=28.3 ns (R²=0.87)
    • Setting dTime = 3σ = 85.0 ns for signal capture

  nPxMin          8            → 5            ↓ (-37.5%) [medium]
    • Small clusters (12.3%) indicate low noise
    • Setting nPxMin = 5

  nPxMax          100          → 120          ↑ (+20.0%) [high]
    • P95 = 98, P99 = 145
    • Setting nPxMax = 120 (P95 × 1.2)
    • Expected rejection rate: 3.2%

PHOTON2EVENT:
----------------------------------------------------------------------
  dSpace_px       50.0         → 58.3         ↑ (+16.6%) [high]
    • High single-photon fraction (32.1%) - parameters too tight
    • Increasing dSpace by 50% to 58.3 px

  dTime_s         5e-08        → 4.2e-08      ↓ (-16.0%) [high]
    • Two-component fit: τ_signal=13.8 ns, τ_background=145.2 ns (R²=0.91)
    • Setting dTime = 3τ_signal = 41.5 ns
    • Estimated purity at this threshold: 73.2%

  durationMax_s   5e-07        → 4.1e-07      ↓ (-18.0%) [high]
    • Exponential fit: τ = 82.3 ns (R²=0.89)
    • P95 = 358.2 ns, P99 = 487.1 ns
    • Setting durationMax = min(5τ, P99) = 411.5 ns
    • Expected rejection rate: 1.8%
======================================================================
```

### Confidence Levels

- **`high`**: Strong statistical evidence, clear distribution features (R² > 0.7 or clear percentiles)
- **`medium`**: Moderate evidence, reasonable fits (R² > 0.5) or indirect indicators
- **`low`**: Insufficient data or poor fits, suggestions are educated guesses

### Change Indicators

- **`↑`**: Parameter should increase (currently too tight)
- **`↓`**: Parameter should decrease (currently too loose)
- **`→`**: Parameter is near optimal (minimal change suggested)

## Detailed Methodology

### 1. Temporal Clustering (`dTime`, `dTime_s`)

**Diagnostic**: Intra-cluster time differences

**What we measure**: For each cluster (photon or event), we compute all pairwise time differences between constituents.

**What we expect**:
- **Signal peak** at small Δt: True correlations from scintillator decay + detector timing
- **Noise floor** at large Δt: Random coincidences following Poisson statistics

**How we optimize**:
1. Fit Gaussian (pixel-to-photon) or two-component exponential (photon-to-event)
2. Extract characteristic timescale (σ for Gaussian, τ for exponential)
3. Set threshold at 3× characteristic scale to capture 99.7% of signal
4. Validate by computing signal/noise ratio in chosen window

**Example**: If pixel time differences show Gaussian with σ=30 ns, set `dTime = 90 ns` (3σ).

### 2. Spatial Clustering (`dSpace`, `dSpace_px`)

**Diagnostic**: Radial distance from cluster centroid

**What we measure**: For each cluster, compute distance of each constituent from the cluster's center of mass.

**What we expect**:
- Exponential or Gaussian decay from center
- 95-99% of constituents within the physical point-spread function
- Small fraction of outliers from noise or pile-up

**How we optimize**:
1. Compute cumulative radial distribution
2. Find P95 and P99 percentiles
3. Calculate outlier fraction with current threshold
4. Adjust to maintain 1-3% outlier rate
   - If outliers > 5%: threshold too tight → increase to P99
   - If outliers < 1%: threshold too loose → decrease to P95

**Example**: If 6% of pixels are beyond current `dSpace = 2.0 mm`, increase to P99 = 2.4 mm.

### 3. Cluster Size Limits (`nPxMin`, `nPxMax`)

**Diagnostic**: Histogram of constituents per cluster

**What we measure**: Distribution of cluster sizes (pixels per photon, photons per event).

**What we expect**:
- Small spike at size=1: Noise or incomplete clustering
- Main peak: Physical cluster size
- Long tail: Bright events or pile-up

**How we optimize**:

**For `nPxMin`**:
1. Compute fraction of small clusters (≤5 constituents)
2. Set threshold to reject noise without cutting signal:
   - If small_fraction > 30%: severe noise → `nPxMin = 8`
   - If small_fraction > 10%: moderate noise → `nPxMin = 5`
   - If small_fraction < 10%: clean data → `nPxMin = 3`
3. Ensure threshold is well below main peak

**For `nPxMax`**:
1. Compute P95 and P99 of cluster size distribution
2. Set at P95 × 1.2 (20% margin)
3. Validate rejection rate < 5%

**Example**: If P95 = 85 pixels, set `nPxMax = 102` to keep 95%+ of clusters.

### 4. Event Duration (`durationMax_s`)

**Diagnostic**: Time span from first to last photon in each event

**What we measure**: For each event, duration = max(photon_times) - min(photon_times).

**What we expect**:
- Exponential decay reflecting scintillator properties
- Typical scale: 10-500 ns depending on scintillator
- Long tail may indicate pile-up or background

**How we optimize**:
1. Fit exponential decay: N(t) = N₀ exp(-t/τ)
2. Extract decay constant τ
3. Compute P95 and P99 percentiles
4. Set threshold conservatively: `durationMax = min(5τ, P99)`
5. Validate rejection rate < 2%

**Example**: If τ = 85 ns and P99 = 450 ns, set `durationMax = min(425, 450) = 425 ns`.

## Typical Parameter Ranges

| Parameter | Typical Range | Depends On |
|-----------|---------------|------------|
| `dTime` | 30-150 ns | Scintillator decay time, detector timing resolution |
| `dSpace` | 1-4 mm | Scintillator thickness, optical point-spread function |
| `nPxMin` | 3-15 pixels | Noise level, pixel size |
| `nPxMax` | 80-200 pixels | Event brightness, pile-up rate |
| `dTime_s` | 20-100 ns | Neutron capture time distribution, event rate |
| `dSpace_px` | 30-80 pixels | Detector magnification, PSF width |
| `durationMax_s` | 200-1000 ns | Scintillator decay time, event complexity |

## When to Use EMPIR Optimization

### ✅ Good Use Cases

1. **New detector setup**: Unknown optimal parameters, need data-driven starting point
2. **Changed conditions**: Different scintillator, lens, or beam characteristics
3. **Iterative refinement**: Have rough parameters, want to fine-tune
4. **Validation**: Check if current parameters are reasonable
5. **Real data without ground truth**: Can't use ground-truth-based optimizer

### ❌ Not Recommended

1. **Insufficient data**: Need at least ~100 clusters for meaningful statistics
2. **Highly contaminated data**: Overwhelming noise makes signal extraction unreliable
3. **You have ground truth**: Use `AssociationOptimizer` with synthetic data instead
4. **Exploratory parameter scan**: This optimizes around current values, doesn't explore full space

## Integration with EMPIR Workflow

### Typical Workflow

```
1. Initial EMPIR reconstruction (with rough parameter guesses)
   ↓
2. Run nea-empir to analyze reconstruction quality
   ↓
3. Review suggested parameters and reasoning
   ↓
4. Update EMPIR parameterSettings.json
   ↓
5. Re-run EMPIR reconstruction with optimized parameters
   ↓
6. (Optional) Iterate if needed
```

### Command Sequence

```bash
# Initial reconstruction (assumed done)
# empir_pixel2photon data.tpx3 --settings initial_params.json
# empir_photon2event photons.empirphot --settings initial_params.json

# Analyze and optimize
nea-empir data_folder/ \
  --current-params initial_params.json \
  --output optimized_params.json \
  -v

# Re-run EMPIR with optimized parameters
# empir_pixel2photon data.tpx3 --settings optimized_params.json
# empir_photon2event photons.empirphot --settings optimized_params.json
```

## Validation Without Ground Truth

Even without association, you can validate parameter quality:

### 1. Reconstruction Efficiency
```python
efficiency = n_pixels_in_photons / n_total_pixels
# Target: > 80% for good parameters
```

### 2. Distribution Quality Metrics
- Signal/noise separation clear (R² > 0.8 for fits)
- No unexpected modes or artifacts in distributions
- Outlier fractions in expected ranges (1-5%)

### 3. Temporal Consistency
- Parameters stable across different time windows
- Similar distributions in early vs. late data

### 4. Physical Consistency
- Parameters match known detector properties
- Event sizes reasonable for neutron imaging
- Duration scales match scintillator decay

## Troubleshooting

### "Insufficient data for analysis"

**Cause**: Too few events/photons for statistical analysis

**Solutions**:
- Load more data files
- Relax initial parameters to get more clusters
- Check if association is working

### Poor fits (R² < 0.5)

**Cause**: Distribution doesn't match expected model

**Solutions**:
- Check for data quality issues
- Review diagnostic plots to understand distribution shape
- Consider manual parameter tuning for this case
- May indicate mixed populations or artifacts

### Confidence levels all "low"

**Cause**: Optimizer uncertain about all parameters

**Solutions**:
- Increase data volume
- Check that correct stage is being optimized
- Verify data has expected structure (pixels grouped into photons, etc.)
- May need manual inspection of distributions

### Suggested parameters seem unreasonable

**Cause**: Optimizer found local minimum or misinterpreted distribution

**Solutions**:
- Check diagnostic metrics in output
- Use `--verbose` to see detailed analysis
- Compare to typical ranges for your setup
- Consider starting from different initial parameters

## Advanced Topics

### Custom Diagnostic Extraction

For specialized analysis, extract diagnostics directly:

```python
import neutron_event_analyzer as nea

# Load and associate data
analyser = nea.Analyse(data_folder='/path/to/data')
analyser.load()
analyser.associate_photons_events()

# Extract diagnostics
diag = nea.EMPIRDiagnostics(
    photons_df=analyser.photons_df,
    events_df=analyser.events_df,
    associated_df=analyser.get_combined_dataframe()
)

# Get specific distributions
time_diffs = diag.intra_photon_time_diffs()
spatial_spread = diag.intra_photon_spatial_spread()
cluster_sizes = diag.photon_cluster_sizes()

# Analyze with custom methods
import matplotlib.pyplot as plt
plt.hist(time_diffs, bins=100)
plt.xlabel('Δt (ns)')
plt.ylabel('Counts')
plt.title('Intra-photon time differences')
plt.show()
```

### Custom Statistical Analysis

Use the distribution analyzer for custom fitting:

```python
from neutron_event_analyzer import DistributionAnalyzer

analyzer = DistributionAnalyzer()

# Fit Gaussian
gaussian_fit = analyzer.fit_gaussian(time_diffs, bins=100)
print(f"μ = {gaussian_fit['mu']:.2f} ns")
print(f"σ = {gaussian_fit['sigma']:.2f} ns")
print(f"R² = {gaussian_fit['r_squared']:.3f}")

# Fit two-component exponential
two_exp_fit = analyzer.fit_two_component_exponential(inter_photon_times)
print(f"Signal τ = {two_exp_fit['tau_signal']:.2f} ns")
print(f"Background τ = {two_exp_fit['tau_background']:.2f} ns")
print(f"Crossover = {two_exp_fit['crossover_point']:.2f} ns")

# Find percentiles
p95 = analyzer.find_percentile_threshold(spatial_spread, 95)
print(f"95th percentile: {p95:.2f} px")
```

## References

### Related Documentation

- [CLI Usage Guide](CLI_USAGE.md) - Complete CLI reference
- [Iterative Optimization](ITERATIVE_OPTIMIZATION.md) - Iterative parameter refinement
- [Optimizer](OPTIMIZER.md) - Ground-truth-based optimization

### Theoretical Background

This framework is based on the principle that reconstruction parameters control specific statistical properties of output distributions. By analyzing these intrinsic signatures, we can identify optimal parameters without external truth information.

Key insight: Good parameters produce distributions with:
1. Clear signal/noise separation (high R² in fits)
2. Minimal outliers (1-5% beyond thresholds)
3. Physical consistency (scales match detector properties)

Poor parameters create:
1. Mixed populations (poor fits, low R²)
2. Excessive outliers (>10%) or no outliers (<1%)
3. Unphysical scales (too large or too small for detector)

---

**Questions or Issues?** Report at https://github.com/TsvikiHirsh/neutron_event_analyzer/issues
