# Association Parameter Optimizer

Automatically find the best association method and parameters for your synthetic data with known ground truth.

## Overview

The `AssociationOptimizer` performs systematic search to find optimal:
- **Association methods** (simple, kdtree, window, lumacam)
- **Spatial distance thresholds** (pixels)
- **Temporal distance thresholds** (nanoseconds)

It evaluates each combination by comparing association results to ground truth, computing metrics like:
- **Association rate**: Fraction of photons associated
- **Accuracy**: Fraction of associations that are correct
- **F1 score**: Harmonic mean of precision and recall

## Quick Start

### Basic Grid Search

```python
from association_optimizer import optimize_for_synthetic_data

# Create synthetic data with ground truth
photon_df = create_synthetic_photon_data(event_configs)  # Has event_id column
event_df = create_synthetic_event_data(event_configs)    # Has event_id column
write_csv_files(None, photon_df, event_df, data_dir)

# Find best parameters
best = optimize_for_synthetic_data(
    synthetic_data_dir=data_dir,
    ground_truth_photons=photon_df,
    ground_truth_events=event_df,
    mode='grid',
    output_dir='results',
    methods=['simple', 'kdtree', 'window'],
    spatial_thresholds_px=[10.0, 20.0, 50.0],
    temporal_thresholds_ns=[50.0, 100.0, 500.0]
)

print(f"Best method: {best.method}")
print(f"Best spatial: {best.spatial_threshold_px} px")
print(f"Best temporal: {best.temporal_threshold_ns} ns")
print(f"F1 Score: {best.f1_score:.4f}")
```

### Recursive Optimization

For fine-tuning existing parameters:

```python
from association_optimizer import AssociationOptimizer

optimizer = AssociationOptimizer(
    synthetic_data_dir=data_dir,
    ground_truth_photons=photon_df,
    ground_truth_events=event_df,
    verbosity=1
)

# Start from existing parameters and optimize
best = optimizer.recursive_optimize(
    method='simple',
    initial_spatial_px=25.0,      # From existing parameterSettings.json
    initial_temporal_ns=150.0,
    max_iterations=10,
    metric='f1_score'
)
```

## Use Cases

### 1. Finding Parameters for New Data Types

When you have a new type of data (e.g., different scintillator, different optics):

```python
# Create representative synthetic data matching your real data characteristics
event_configs = [{
    'event_id': i,
    'center_x': ...,
    'center_y': ...,
    't_ns': ...,
    'n_photons': ...,
    'photon_spread_spatial': ...,  # Match your real data
    'photon_spread_temporal': ...
} for i in range(10)]

# Optimize
best = optimize_for_synthetic_data(...)

# Save parameters
optimizer.save_best_parameters('my_new_params.json')
```

### 2. Validating Existing Parameters

Check if your current `parameterSettings.json` is optimal:

```python
# Load existing parameters
with open('parameterSettings.json') as f:
    existing = json.load(f)

current_spatial = existing['photon2event']['dSpace_px']
current_temporal = existing['photon2event']['dTime_s'] * 1e9

# Evaluate current performance
current_result = optimizer.evaluate_association(
    method='simple',
    spatial_threshold_px=current_spatial,
    temporal_threshold_ns=current_temporal
)

# Try to improve
best_result = optimizer.recursive_optimize(
    method='simple',
    initial_spatial_px=current_spatial,
    initial_temporal_ns=current_temporal
)

print(f"Current F1: {current_result.f1_score:.4f}")
print(f"Optimized F1: {best_result.f1_score:.4f}")
print(f"Improvement: {(best_result.f1_score - current_result.f1_score):.4f}")
```

### 3. Comparing Association Methods

Find which method works best for your data:

```python
results = {}
for method in ['simple', 'kdtree', 'window']:
    optimizer = AssociationOptimizer(...)
    best = optimizer.grid_search(
        methods=[method],
        spatial_thresholds_px=[10.0, 20.0, 50.0],
        temporal_thresholds_ns=[50.0, 100.0, 500.0]
    )
    results[method] = best

# Compare
for method, result in results.items():
    print(f"{method}: F1={result.f1_score:.4f}, "
          f"Spatial={result.spatial_threshold_px}px, "
          f"Temporal={result.temporal_threshold_ns}ns")
```

## Optimization Modes

### Grid Search (`mode='grid'`)

**When to use:**
- Initial exploration of parameter space
- Comparing multiple methods
- Need comprehensive coverage

**Advantages:**
- Tests all combinations systematically
- Guaranteed to find best within grid
- Can compare very different parameter ranges

**Disadvantages:**
- Can be slow for large parameter spaces
- Might miss optimal values between grid points

**Example:**
```python
best = optimizer.grid_search(
    methods=['simple', 'kdtree', 'window'],
    spatial_thresholds_px=[5.0, 10.0, 20.0, 50.0, 100.0],
    temporal_thresholds_ns=[20.0, 50.0, 100.0, 200.0, 500.0],
    metric='f1_score'
)
# Tests 3 × 5 × 5 = 75 combinations
```

### Recursive Optimization (`mode='recursive'`)

**When to use:**
- Fine-tuning existing parameters
- Continuous optimization in parameter space
- Need faster convergence

**Advantages:**
- Faster than grid search
- Finds local optimum efficiently
- Can fine-tune to precise values

**Disadvantages:**
- May converge to local optimum (not global)
- Depends on good starting point

**Example:**
```python
best = optimizer.recursive_optimize(
    method='simple',
    initial_spatial_px=20.0,
    initial_temporal_ns=100.0,
    max_iterations=10,
    convergence_threshold=0.001,
    metric='f1_score'
)
# Iteratively improves from starting point
```

## Optimization Metrics

### F1 Score (`metric='f1_score'`) - **Recommended**

Harmonic mean of precision and recall. Balances:
- **Precision** (accuracy): Are associations correct?
- **Recall** (association rate): Are photons being associated?

**Use when:** You want balanced performance - both correct associations AND high association rate.

### Accuracy (`metric='accuracy'`)

Fraction of associations that are correct.

**Use when:** Correctness is more important than coverage. Better to miss associations than make wrong ones.

### Association Rate (`metric='association_rate'`)

Fraction of photons that get associated.

**Use when:** Coverage is more important than precision. Want to associate as many photons as possible.

## Ground Truth Format

The optimizer compares results to ground truth DataFrames:

### Ground Truth Photons

Must have these columns:
```python
photon_df = pd.DataFrame({
    'x': [...],          # Photon x position
    'y': [...],          # Photon y position
    'toa_ns': [...],     # Time of arrival (ns)
    'tof_ns': [...],     # Time of flight
    'event_id': [...]    # Ground truth event ID ← REQUIRED
})
```

### Ground Truth Events

Must have these columns:
```python
event_df = pd.DataFrame({
    'event_id': [...],   # Event ID
    'center_x': [...],   # Event center x
    'center_y': [...],   # Event center y
    't_ns': [...],       # Event time (ns)
    'n': [...]           # Number of photons (optional)
})
```

The optimizer uses `event_id` to verify if photons were associated to the correct events.

## Output Files

### optimization_results.json

Complete results from all tested combinations:

```json
{
  "best_result": {
    "method": "simple",
    "spatial_threshold_px": 20.0,
    "temporal_threshold_ns": 100.0,
    "total_photons": 15,
    "associated_photons": 14,
    "correctly_associated": 14,
    "association_rate": 0.933,
    "accuracy": 1.0,
    "f1_score": 0.965,
    "unique_events_found": 3,
    "expected_events": 3,
    "avg_com_distance": 5.2
  },
  "all_results": [
    {...},
    {...}
  ]
}
```

### best_parameters.json

Best parameters in empir `parameterSettings.json` format:

```json
{
  "photon2event": {
    "dSpace_px": 20.0,
    "dTime_s": 1e-07,
    "durationMax_s": 1e-06,
    "dTime_ext": 5
  }
}
```

You can directly use this file with empir or NEA.

## Integration with G4LumaCam Workflow

Complete workflow for empir validation:

```python
# 1. Create synthetic pixel data
pixel_df = create_synthetic_pixel_data(photon_configs)

# 2. Write to TPX3
write_tpx3_file(pixel_df, 'synthetic.tpx3')

# 3. Process through empir
from lumacam import Analysis
analysis = Analysis(archive='test_dir')
analysis.process(
    params='fast_neutrons',
    pixel2photon=True,
    photon2event=True,
    export_photons=True,
    export_events=True
)

# 4. Optimize NEA association parameters
best = optimize_for_synthetic_data(
    synthetic_data_dir='test_dir',
    ground_truth_photons=photon_df,
    ground_truth_events=event_df,
    mode='grid'
)

# 5. Save optimized parameters
optimizer.save_best_parameters('optimized_params.json')

# 6. Use optimized parameters with empir
analysis.process(params='optimized_params.json', ...)
```

## Advanced Usage

### Custom Metrics

You can add custom metrics by extending `AssociationMetrics`:

```python
@dataclass
class CustomMetrics(AssociationMetrics):
    custom_score: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # Your custom metric calculation
        self.custom_score = self.accuracy * 2 + self.association_rate

# Then optimize for it
best = optimizer.grid_search(metric='custom_score')
```

### Parameter Ranges Based on Data

Automatically set parameter ranges based on data characteristics:

```python
# Analyze synthetic data
spatial_std = np.std([
    np.sqrt((p['x'] - e['center_x'])**2 + (p['y'] - e['center_y'])**2)
    for p in photon_df.itertuples()
    for e in event_df.itertuples()
    if p.event_id == e.event_id
])

temporal_std = photon_df.groupby('event_id')['toa_ns'].std().mean()

# Set ranges based on data
spatial_range = [0.5 * spatial_std, 3.0 * spatial_std]
temporal_range = [0.5 * temporal_std, 3.0 * temporal_std]

# Optimize
best = optimizer.recursive_optimize(
    method='simple',
    initial_spatial_px=spatial_std,
    initial_temporal_ns=temporal_std,
    spatial_range=spatial_range,
    temporal_range=temporal_range
)
```

### Parallel Optimization

For very large parameter spaces, you can parallelize:

```python
from concurrent.futures import ProcessPoolExecutor

def optimize_chunk(methods_chunk):
    optimizer = AssociationOptimizer(...)
    return optimizer.grid_search(methods=methods_chunk, ...)

# Split work
chunks = [['simple'], ['kdtree'], ['window']]

with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(optimize_chunk, chunks))

# Find overall best
best = max(results, key=lambda r: r.f1_score)
```

## Examples

See `example_optimizer_usage.py` for complete working examples:

1. **Basic grid search** - Find best method and parameters
2. **Recursive optimization** - Fine-tune from starting point
3. **Save results** - Export results to files
4. **Optimize from existing params** - Improve existing parameterSettings.json
5. **Compare metrics** - See how different objectives affect results

Run all examples:
```bash
python tests/example_optimizer_usage.py
```

## API Reference

### AssociationOptimizer

```python
optimizer = AssociationOptimizer(
    synthetic_data_dir: str,           # Directory with data files
    ground_truth_photons: pd.DataFrame, # Ground truth photons with event_id
    ground_truth_events: pd.DataFrame,  # Ground truth events
    verbosity: int = 1                  # 0=silent, 1=normal, 2=detailed
)
```

**Methods:**

- `evaluate_association(method, spatial_threshold_px, temporal_threshold_ns, settings=None)` → `AssociationMetrics`
- `grid_search(methods, spatial_thresholds_px, temporal_thresholds_ns, settings=None, metric='f1_score')` → `AssociationMetrics`
- `recursive_optimize(method, initial_spatial_px, initial_temporal_ns, max_iterations=10, metric='f1_score')` → `AssociationMetrics`
- `save_results(output_path)` - Save all results to JSON
- `save_best_parameters(output_path)` - Save best params as parameterSettings.json
- `get_best_parameters_json()` → Dict - Get best params as dictionary

### AssociationMetrics

```python
@dataclass
class AssociationMetrics:
    method: str                    # Association method used
    spatial_threshold_px: float    # Spatial threshold (pixels)
    temporal_threshold_ns: float   # Temporal threshold (ns)
    total_photons: int             # Total photons in dataset
    associated_photons: int        # Number associated
    correctly_associated: int      # Number correctly associated
    association_rate: float        # Fraction associated (0-1)
    accuracy: float                # Fraction correct (0-1)
    f1_score: float               # Harmonic mean (0-1)
    unique_events_found: int       # Events found
    expected_events: int           # Expected events
    avg_com_distance: float        # Avg center-of-mass distance
```

### Convenience Function

```python
best = optimize_for_synthetic_data(
    synthetic_data_dir: str,
    ground_truth_photons: pd.DataFrame,
    ground_truth_events: pd.DataFrame,
    mode: str = 'grid',               # 'grid' or 'recursive'
    output_dir: str = None,           # Save results here
    verbosity: int = 1,
    **kwargs                          # Passed to optimizer
) → AssociationMetrics
```

## Tips & Best Practices

### 1. Start with Grid Search

Begin with a coarse grid to explore the parameter space:

```python
# Coarse grid
best_coarse = optimizer.grid_search(
    methods=['simple', 'kdtree'],
    spatial_thresholds_px=[10.0, 50.0, 100.0],
    temporal_thresholds_ns=[50.0, 500.0]
)

# Then fine-tune with recursive
best_fine = optimizer.recursive_optimize(
    method=best_coarse.method,
    initial_spatial_px=best_coarse.spatial_threshold_px,
    initial_temporal_ns=best_coarse.temporal_threshold_ns
)
```

### 2. Match Synthetic Data to Real Data

Your synthetic data should match real data characteristics:

```python
# Analyze real data first
real_spatial_spread = ...  # From real data analysis
real_temporal_spread = ...

# Create matching synthetic data
event_configs = [{
    'photon_spread_spatial': real_spatial_spread,
    'photon_spread_temporal': real_temporal_spread,
    ...
}]
```

### 3. Test Multiple Scenarios

Optimize for different scenarios:

```python
scenarios = {
    'tight_clustering': {...},
    'loose_clustering': {...},
    'temporal_proximity': {...}
}

for name, config in scenarios.items():
    photons = create_synthetic_photon_data(config)
    events = create_synthetic_event_data(config)
    best = optimize_for_synthetic_data(...)
    print(f"{name}: {best}")
```

### 4. Validate on Hold-Out Data

Reserve some synthetic data for validation:

```python
# Split data
train_photons, test_photons = train_test_split(photon_df)
train_events, test_events = train_test_split(event_df)

# Optimize on training data
optimizer_train = AssociationOptimizer(..., ground_truth_photons=train_photons)
best = optimizer_train.grid_search(...)

# Validate on test data
optimizer_test = AssociationOptimizer(..., ground_truth_photons=test_photons)
test_result = optimizer_test.evaluate_association(
    method=best.method,
    spatial_threshold_px=best.spatial_threshold_px,
    temporal_threshold_ns=best.temporal_threshold_ns
)

print(f"Training F1: {best.f1_score:.4f}")
print(f"Test F1: {test_result.f1_score:.4f}")
```

## Troubleshooting

### No associations found

- Check that spatial/temporal ranges include reasonable values
- Verify ground truth data has correct format
- Check that synthetic data was written correctly

### All metrics are 0

- Verify CSV files exist in expected directories
- Check that dummy binary files were created
- Run with `verbosity=2` to see detailed progress

### Low accuracy despite high association rate

- Synthetic data may not match real data characteristics
- Association parameters may be too loose (associating wrong photons)
- Try tighter spatial/temporal thresholds

### Optimization converges too quickly

- Increase grid resolution
- Use recursive optimization with smaller step sizes
- Check convergence_threshold isn't too large

## References

- Association algorithms: `src/neutron_event_analyzer/analyser.py`
- Configuration presets: `src/neutron_event_analyzer/config.py`
- Synthetic data generation: `tests/test_association_validation.py`
- Usage examples: `tests/example_optimizer_usage.py`
