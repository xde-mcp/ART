# SWE-bench Test Runner Status Summary

## Current Statistics (Instances 0-399)

- **Total Tested**: 369 instances (31 filtered out from 400 total)
- **Successful**: 344 instances (93.2%)
- **Blacklisted**: 25 instances (6.8%)

## Generic Solutions Implemented

### 1. Missing Pytest Plugins Detection

- **Problem**: pytest fails with "Missing required plugins: pytest-cov, pytest-xdist"
- **Solution**: Automatically detects and installs required plugins
- **Impact**: Fixed instances like HIPS\_\_autograd that required pytest-cov

### 2. Recursive Optional Dependencies

- **Problem**: Cascading optional dependencies (e.g., sunpy[timeseries] → sunpy[visualization])
- **Solution**: Recursively installs suggested packages from conftest errors
- **Impact**: Fixed sunpy instances that previously failed

### 3. Enhanced Module Installation

- **Problem**: Missing Python packages with non-standard names
- **Solution**: Tries multiple package name variants (e.g., jwt → PyJWT)
- **Impact**: Improved success rate for packages with naming mismatches

## Blacklisted Instance Categories

### 1. Pandas Instances (10 total)

- **Issue**: Missing compiled C extensions (pandas.\_libs.pandas_parser)
- **Why Blacklisted**: Requires building pandas from source with C compilation
- **New in 300-399**: pandas-dev__pandas.95280573.pr_48426, pandas-dev__pandas.95280573.pr_48966

### 2. PyPika Instances (6 total)

- **Issue**: No tests collected (0 items)
- **Why Blacklisted**: Fundamental test discovery incompatibility
- **New in 300-399**: kayak__pypika.1c9646f0.lm_rewrite__xdfkt9wb, kayak__pypika.1c9646f0.func_pm_class_rm_funcs__muwfrdsf

### 3. Patch Issues (5 total)

- **Issue**: Tests pass when they should fail (patch doesn't introduce bug)
- **Why Blacklisted**: Problem with the instance itself, not the test runner

### 4. Django-Money Instances (1 total)

- **Issue**: No tests executed
- **Why Blacklisted**: Test collection issue similar to PyPika
- **New pattern in 300-399**: django-money__django-money.835c1ab8.lm_rewrite__bv3wb2wq

### 5. Other (3 total)

- **Issue**: Various unique issues including timeouts
- **Why Blacklisted**: Would require case-specific fixes or timeout adjustments

## Key Achievements

1. **High Success Rate**: 93.2% of instances work without case-specific code
2. **Generic Solutions**: All fixes are pattern-based, not instance-specific
3. **No Performance Regression**: Working instances continue to work efficiently (verified with regression tests)
4. **Clear Failure Patterns**: Identified and documented common failure types
5. **Automated Recovery**: Test runner automatically retries with fixes

## Test Results Summary

### Instances 0-299
- **Success Rate**: 94% (282/300)
- **Blacklisted**: 18 instances

### Instances 300-399
- **Success Rate**: 89.9% (62/69 tested)
- **New Blacklisted**: 7 instances
- **New Pattern**: Django-Money test collection issues

## Next Instance Batches to Test

- 400-499
- 500-599
- ... (total of ~8600 instances available)

## Code Quality

- Clean, maintainable code with clear comments
- Diagnostic messages only when needed
- All blacklisted instances have documented reasons
- Generic solutions handle patterns, not specific cases
