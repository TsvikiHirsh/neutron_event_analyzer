# Iterative Parameter Optimization

Automatically improve association parameters by analyzing real data through multiple iterations.

## Overview

The iterative optimizer helps you find better association parameters without needing ground truth data. It:

1. **Runs association** with current parameters
2. **Analyzes quality metrics** (association rate, event sizes, spreads, etc.)
3. **Suggests improvements** based on data characteristics
4. **Repeats** until convergence or max iterations

This is perfect for tuning parameters on real TPX3 data where you don't have ground truth.

## Quick Start

### Command Line

```bash
# Basic usage - optimize with 5 iterations
nea-optimize optimize /path/to/data --output results/

# Custom initial parameters
nea-optimize optimize /path/to/data \
  --spatial 25 \
  --temporal 150 \
  --iterations 10 \
  --output results/

# Quick suggestion (single iteration)
nea-optimize suggest /path/to/data \
  --spatial 20 \
  --temporal 100 \
  --output suggested_params.json
```

### Python API

```python
import neutron_event_analyzer as nea

# Quick optimization
best_params = nea.optimize_parameters_iteratively(
    data_folder='path/to/data',
    initial_spatial_px=20.0,
    initial_temporal_ns=100.0,
    max_iterations=5,
    output_dir='results/'
)

print(f"Best spatial: {best_params['spatial_px']:.2f} px")
print(f"Best temporal: {best_params['temporal_ns']:.2f} ns")
```

## CLI Commands

### `optimize` - Iterative Optimization

Run multiple iterations to progressively refine parameters.

```bash
nea-optimize optimize DATA_FOLDER [OPTIONS]
```

**Arguments:**
- `DATA_FOLDER`: Directory containing photon/event CSV files

**Options:**
- `-s, --spatial FLOAT`: Initial spatial threshold in pixels (default: 20.0)
- `-t, --temporal FLOAT`: Initial temporal threshold in nanoseconds (default: 100.0)
- `-n, --iterations INT`: Maximum number of iterations (default: 5)
- `-c, --convergence FLOAT`: Convergence threshold (default: 0.05)
- `-m, --method {simple,kdtree,window,lumacam}`: Association method (default: simple)
- `--settings PATH`: Settings preset or file path
- `-o, --output DIR`: Output directory for results
- `-v, --verbose`: Increase verbosity (use `-vv` for detailed output)

**Example:**
```bash
nea-optimize optimize ~/data/neutrons \
  --spatial 25 \
  --temporal 150 \
  --iterations 10 \
  --method kdtree \
  --output ~/results/optimization \
  -vv
```

**Output Files:**
- `best_parameters.json`: Optimized parameterSettings.json
- `optimization_history.json`: Full iteration history
- `summary.json`: Optimization summary

### `suggest` - Single Suggestion

Analyze data once and suggest improved parameters.

```bash
nea-optimize suggest DATA_FOLDER [OPTIONS]
```

**Arguments:**
- `DATA_FOLDER`: Directory containing data files

**Options:**
- `-s, --spatial FLOAT`: Current spatial threshold (default: 20.0)
- `-t, --temporal FLOAT`: Current temporal threshold (default: 100.0)
- `-m, --method {simple,kdtree,window,lumacam}`: Association method (default: simple)
- `--settings PATH`: Settings preset or file path
- `-o, --output FILE`: Output JSON file for suggested parameters
- `-v, --verbose`: Increase verbosity

**Example:**
```bash
nea-optimize suggest ~/data/neutrons \
  --spatial 20 \
  --temporal 100 \
  --output next_params.json
```

### `analyze` - Quality Analysis

Analyze association quality without suggesting changes.

```bash
nea-optimize analyze DATA_FOLDER [OPTIONS]
```

**Arguments:**
- `DATA_FOLDER`: Directory containing data files

**Options:**
- `-s, --spatial FLOAT`: Spatial threshold (default: 20.0)
- `-t, --temporal FLOAT`: Temporal threshold (default: 100.0)
- `-m, --method {simple,kdtree,window,lumacam}`: Association method
- `--settings PATH`: Settings preset or file path
- `-o, --output FILE`: Output JSON file for metrics
- `-v, --verbose`: Increase verbosity

**Example:**
```bash
nea-optimize analyze ~/data/neutrons \
  --spatial 20 \
  --temporal 100 \
  --output metrics.json
```

## Python API

### IterativeOptimizer Class

```python
from neutron_event_analyzer import IterativeOptimizer

# Initialize
optimizer = IterativeOptimizer(
    data_folder='path/to/data',
    initial_spatial_px=20.0,
    initial_temporal_ns=100.0,
    settings=None,
    method='simple',
    verbosity=1
)

# Run optimization
best_result = optimizer.optimize(
    max_iterations=5,
    convergence_threshold=0.05,
    output_dir='results/'
)

# Access results
print(f"Best spatial: {best_result.spatial_px:.2f} px")
print(f"Best temporal: {best_result.temporal_ns:.2f} ns")
print(f"Association rate: {best_result.association_rate:.2%}")

# Get progress as DataFrame
df = optimizer.get_progress_dataframe()
print(df)

# Plot progress
optimizer.plot_progress(save_path='optimization_progress.png')
```

### ParameterSuggester Class

For single-shot parameter suggestions:

```python
from neutron_event_analyzer import ParameterSuggester
import neutron_event_analyzer as nea

# Load and associate data
analyser = nea.Analyse(data_folder='path/to/data')
analyser.load()
analyser.associate_photons_events(method='simple', dSpace_px=20, max_time_ns=100)

# Analyze and suggest
suggester = ParameterSuggester(analyser, verbosity=1)
metrics = suggester.analyze_quality()
suggestion = suggester.suggest_parameters(
    current_spatial_px=20.0,
    current_temporal_ns=100.0
)

# Save suggestions
suggester.save_suggested_parameters('next_params.json')
```

## How It Works

### Quality Metrics Analyzed

The optimizer examines:

1. **Association Rate**: Fraction of photons successfully associated
   - Low rate → parameters might be too tight
   - Very high rate → might be over-associating

2. **Event Sizes**: Distribution of photons per event
   - Very large events → parameters too loose
   - Many tiny events → parameters too tight

3. **Spatial Spread**: How far photons are from event centers
   - Suggests appropriate spatial thresholds

4. **Temporal Spread**: Time distribution within events
   - Suggests appropriate temporal thresholds

5. **Outlier Detection**: Identifies photons far from event centers
   - Indicates parameter issues

### Decision Logic

The suggester uses heuristics to adjust parameters:

**Too Tight (Under-associated):**
- Association rate < 30% → Increase thresholds by 50%
- Association rate < 60% → Increase thresholds by 20%

**Too Loose (Over-associated):**
- Very large events (>20 photons) → Decrease by 30%
- Large spreads or outliers → Decrease by 10-15%

**Adaptive Adjustment:**
- Target thresholds ~2x the median spatial/temporal spread
- Converges when changes < 5% between iterations

## Use Cases

### 1. Initial Parameter Discovery

Starting from scratch with new data:

```bash
# Use default starting values
nea-optimize optimize ~/new_data \
  --iterations 10 \
  --output ~/results
```

### 2. Fine-Tuning Existing Parameters

You have parameters but want to improve:

```bash
# Start from your current parameters
nea-optimize optimize ~/data \
  --spatial 35 \
  --temporal 180 \
  --iterations 5 \
  --output ~/improved
```

### 3. Quick Parameter Check

Verify if current parameters are reasonable:

```bash
nea-optimize suggest ~/data \
  --spatial 20 \
  --temporal 100
```

### 4. Comparing Methods

Find which association method works best:

```bash
for method in simple kdtree window; do
  nea-optimize optimize ~/data \
    --method $method \
    --output ~/results_$method
done
```

## Workflow Example

Complete workflow for parameter optimization:

```bash
# 1. Start with analysis of current parameters
nea-optimize analyze ~/data/run001 \
  --spatial 20 \
  --temporal 100 \
  --output initial_metrics.json

# 2. Run iterative optimization
nea-optimize optimize ~/data/run001 \
  --spatial 20 \
  --temporal 100 \
  --iterations 5 \
  --output ~/optimization_results \
  -vv

# 3. Check the results
cat ~/optimization_results/summary.json

# 4. Use optimized parameters in your workflow
cp ~/optimization_results/best_parameters.json ~/my_analysis/parameterSettings.json

# 5. Validate on another dataset
nea-optimize analyze ~/data/run002 \
  --spatial $(jq -r '.photon2event.dSpace_px' ~/optimization_results/best_parameters.json) \
  --temporal $(jq -r '.photon2event.dTime_s * 1e9' ~/optimization_results/best_parameters.json)
```

## Integration with empir

If you're using empir to process TPX3 files:

```bash
# 1. Process TPX3 through empir
lumacam process raw_data/ \
  --params initial_params.json \
  --pixel2photon \
  --photon2event \
  --export-photons \
  --export-events

# 2. Optimize NEA parameters on empir output
nea-optimize optimize raw_data/ \
  --iterations 5 \
  --output optimized/

# 3. Create new empir params with optimized values
# (update the photon2event section in your empir params)

# 4. Reprocess with better parameters
lumacam process raw_data/ \
  --params optimized/best_parameters.json \
  --photon2event \
  --export-events
```

## Output Format

### best_parameters.json

Standard parameterSettings.json format:

```json
{
  "photon2event": {
    "dSpace_px": 23.45,
    "dTime_s": 1.2e-07,
    "durationMax_s": 1.2e-06,
    "dTime_ext": 5
  }
}
```

### optimization_history.json

Full iteration history:

```json
[
  {
    "iteration": 1,
    "spatial_px": 20.0,
    "temporal_ns": 100.0,
    "method": "simple",
    "association_rate": 0.65,
    "mean_photons_per_event": 8.3,
    "total_events": 145
  },
  {
    "iteration": 2,
    "spatial_px": 22.5,
    "temporal_ns": 110.0,
    ...
  }
]
```

### summary.json

Optimization summary:

```json
{
  "initial_parameters": {
    "spatial_px": 20.0,
    "temporal_ns": 100.0
  },
  "best_parameters": {
    "spatial_px": 23.45,
    "temporal_ns": 115.67
  },
  "improvement": {
    "spatial_change_pct": 17.25,
    "temporal_change_pct": 15.67
  },
  "total_iterations": 4,
  "best_iteration": 3,
  "final_association_rate": 0.82,
  "final_mean_photons_per_event": 7.5
}
```

## Tips & Best Practices

### 1. Start Conservative

Begin with moderate parameters and let the optimizer adjust:

```bash
nea-optimize optimize data/ \
  --spatial 20 \
  --temporal 100  # Safe starting values
```

### 2. Use Multiple Datasets

Validate optimized parameters on different runs:

```bash
# Optimize on one dataset
nea-optimize optimize data/run001 --output opt1/

# Validate on others
nea-optimize analyze data/run002 --spatial $(cat opt1/best_params...)
nea-optimize analyze data/run003 --spatial $(cat opt1/best_params...)
```

### 3. Check Convergence

If not converging, increase iterations or check data quality:

```bash
nea-optimize optimize data/ \
  --iterations 10 \
  --convergence 0.02  # Tighter convergence
```

### 4. Compare Methods

Different methods may work better for different data:

```bash
for method in simple kdtree window; do
  nea-optimize optimize data/ --method $method --output results_$method/
done
```

### 5. Iterative Refinement

Use output as input for next optimization:

```bash
# First pass
nea-optimize optimize data/ --output pass1/

# Second pass with refined starting point
nea-optimize optimize data/ \
  --spatial $(jq -r '.photon2event.dSpace_px' pass1/best_parameters.json) \
  --temporal $(jq -r '.photon2event.dTime_s * 1e9' pass1/best_parameters.json) \
  --output pass2/
```

## Troubleshooting

### No improvement after iterations

- Data might be difficult to cluster
- Try different association methods
- Check if data quality is good
- Manually inspect photon/event distributions

### Parameters diverging

- Initial parameters might be very far from optimal
- Try more conservative starting values
- Reduce convergence threshold
- Check for data issues (noise, artifacts)

### Low association rates persist

- Data might have high noise
- Events might be very sparse
- Consider data preprocessing
- Try looser initial parameters

## References

- Parameter Suggester: [src/neutron_event_analyzer/parameter_suggester.py](../src/neutron_event_analyzer/parameter_suggester.py)
- Iterative Optimizer: [src/neutron_event_analyzer/iterative_optimizer.py](../src/neutron_event_analyzer/iterative_optimizer.py)
- CLI Implementation: [src/neutron_event_analyzer/cli.py](../src/neutron_event_analyzer/cli.py)
- Optimizer (with ground truth): [OPTIMIZER.md](OPTIMIZER.md)
