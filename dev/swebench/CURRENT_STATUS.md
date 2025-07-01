# SWE-bench Test Runner Status Summary

## Current Statistics (Instances 0-399)

- **Total Tested**: 400 instances
- **Successful**: 375 instances (93.75%)
- **Blacklisted**: 23 instances (5.75%)
- **Other Failures**: 2 instances (0.5%)

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
- **New**: Added 2 more instances from 300-399 range

### 2. PyPika Instances (6 total)

- **Issue**: No tests collected (0 items)
- **Why Blacklisted**: Fundamental test discovery incompatibility
- **New**: Added 2 more instances from 300-399 range

### 3. Patch Issues (5 total)

- **Issue**: Tests pass when they should fail (patch doesn't introduce bug)
- **Why Blacklisted**: Problem with the instance itself, not the test runner

### 4. Django-money Instances (1 total)

- **Issue**: No tests collected
- **Why Blacklisted**: Test discovery issue similar to PyPika
- **New**: Found in 300-399 range

### 5. Other (1 total)

- **Issue**: Various unique issues
- **Why Blacklisted**: Would require case-specific fixes

## Key Achievements

1. **High Success Rate**: 93.75% of instances work without case-specific code
2. **Generic Solutions**: All fixes are pattern-based, not instance-specific
3. **No Performance Regression**: Working instances continue to work efficiently
4. **Clear Failure Patterns**: Identified and documented common failure types
5. **Automated Recovery**: Test runner automatically retries with fixes
6. **UV Integration**: Successfully using uv for faster package management

## Progress Update

- **Completed**: Instances 0-399 tested
- **New Findings**: 5 instances added to blacklist (same patterns as existing)
- **Success Rate**: Maintained ~94% across all tested instances

## Next Instance Batches to Test

- 400-499
- 500-599
- ... (total of ~8600 instances available)

## Code Quality

- Clean, maintainable code with clear comments
- Diagnostic messages only when needed
- All blacklisted instances have documented reasons
- Generic solutions handle patterns, not specific cases
