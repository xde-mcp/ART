# Boolean Support for Trajectory Metrics

## Summary

Boolean support has been added to trajectory metrics in the ART framework. Boolean values (`True`/`False`) are now properly converted to floats (`1.0`/`0.0`) when calculating averages, ensuring accurate metric aggregation.

## Changes Made

### 1. Updated `src/art/gather.py`

**File**: `src/art/gather.py`  
**Function**: `record_metrics()`  
**Change**: Modified metric aggregation to convert boolean values to floats before adding to metric sums.

**Before**:
```python
context.metric_sums.update(trajectory.metrics)
```

**After**:
```python
# Convert boolean values to floats before adding to metric_sums
for metric, value in trajectory.metrics.items():
    context.metric_sums[metric] += float(value)
```

### 2. Updated `src/art/utils/old_benchmarking/load_benchmarked_models.py`

**File**: `src/art/utils/old_benchmarking/load_benchmarked_models.py`  
**Function**: `load_benchmarked_models()`  
**Change**: Added explicit float conversion when calculating metric averages.

**Before**:
```python
average = sum(
    trajectory.metrics[metric]
    for trajectory in trajectories_with_metric
) / len(trajectories_with_metric)
```

**After**:
```python
# Convert boolean values to floats before averaging
average = sum(
    float(trajectory.metrics[metric])
    for trajectory in trajectories_with_metric
) / len(trajectories_with_metric)
```

## Existing Support

The following components already had proper boolean support:

- **`src/art/trajectories.py`**: The `Trajectory` class already included `bool` in the metrics type annotation: `metrics: dict[str, float | int | bool] = {}`
- **`src/art/local/backend.py`**: The `_log()` method already converts values with `float(value)` before aggregation

## Boolean Conversion Rules

- `True` → `1.0`
- `False` → `0.0`
- Existing `int` and `float` values remain unchanged
- Mixed boolean and numeric metrics in the same trajectory are properly supported

## Usage Examples

You can now use boolean metrics in trajectories:

```python
trajectory = Trajectory(
    messages_and_choices=[...],
    reward=1.0,
    metrics={
        "success": True,           # Will be converted to 1.0 for averaging
        "has_error": False,        # Will be converted to 0.0 for averaging
        "is_complete": True,       # Will be converted to 1.0 for averaging
        "score": 0.85,            # Float values work as before
        "attempts": 3,            # Integer values work as before
    }
)
```

When averages are calculated across multiple trajectories:
- Boolean metrics will be averaged as floats (e.g., 2 `True` + 1 `False` = average of 0.667)
- The resulting averages will always be floats between 0.0 and 1.0 for boolean metrics

## Verification

The implementation has been tested to ensure:
1. Boolean values are properly converted to floats during metric aggregation
2. Averages are calculated correctly across multiple trajectories
3. Mixed boolean, integer, and float metrics work together seamlessly
4. No breaking changes to existing functionality

This enhancement maintains full backward compatibility while enabling more intuitive boolean metric usage in trajectory evaluation systems.