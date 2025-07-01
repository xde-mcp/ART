# Instructions for Continuing SWE-bench Test Runner Improvements

## Overview

This document provides instructions for continuing work on improving `daytona.py` to maximize the number of SWE-bench instances that can successfully run their tests.

## Current State (as of testing instances 0-299)

- **Success Rate**: 94% (282/300 instances working)
- **Blacklisted**: 18 instances (6%) with documented technical reasons
- **Key Files**:
  - `daytona.py` - Main test runner with generic fixes
  - `instances.py` - Contains blacklist of problematic instances

## Goals

1. **Maximize Success Rate**: Get as many instances working as possible
2. **Generic Solutions Only**: Avoid case-specific code; implement solutions that address patterns
3. **Performance**: No regression on already-working instances
4. **Clear Documentation**: Blacklist only when generic solutions aren't feasible

## 🚨 Critical Requirements - MUST READ

### 1. Zero Regression Policy

**Every change must maintain 100% success rate on previously working instances.**

Before merging ANY change:

```bash
# Test a sample of known-working instances
python daytona.py 0-9 --parallel --print-exceptions --use-pbar

# If all pass, test a larger set
python daytona.py 0-49 --parallel --print-exceptions --use-pbar

# For major changes, test all previously working instances
python daytona.py 0-99 --parallel --print-exceptions --use-pbar
```

If ANY previously working instance fails after your change, DO NOT proceed. Either fix the regression or revert your change.

### 2. Fundamental Test Guarantee

**An instance is only considered "working" if:**

- All FAIL_TO_PASS tests actually fail (the patch introduced the bug correctly)
- All PASS_TO_PASS tests actually pass (existing functionality still works)

This is verified by the `analyze_test_results` function and the message "✓ Tests are failing as expected".

**Never compromise this guarantee**. If you can get tests to run but they don't show the expected behavior, the instance should be blacklisted.

### 3. Code Reduction Opportunities

While adding new solutions, always look for opportunities to:

- **Consolidate similar patterns** into single solutions
- **Remove redundant code** that handles the same issue differently
- **Simplify complex logic** while maintaining functionality

Example of code reduction:

```python
# Instead of multiple specific checks:
if "pytest-cov" in error:
    install("pytest-cov")
if "pytest-xdist" in error:
    install("pytest-xdist")
if "pytest-timeout" in error:
    install("pytest-timeout")

# Use a generic pattern:
if "Missing required plugins:" in error:
    plugins = extract_plugins(error)
    for plugin in plugins:
        install(plugin)
```

## Methodology

### 1. Testing New Instances

```bash
# Test the next batch of 100 instances
cd dev/swebench
python daytona.py 300-399 --parallel --print-exceptions --use-pbar 2>&1 | tee test_run_300_399.log
```

### 2. Analyzing Failures

Look for patterns in the log files:

```bash
# Find instances that failed to run tests
grep "WARNING: No tests were executed|No tests executed for" test_run_300_399.log

# Find common error patterns
grep "ERROR: not found:|ERROR: found no collectors|Missing required plugins:" test_run_300_399.log

# Count successes
grep "Ready for agent" test_run_300_399.log | wc -l

# IMPORTANT: Verify test behavior is correct
grep "Tests are failing as expected" test_run_300_399.log | wc -l
```

### 3. Implementing Generic Solutions

#### Current Generic Solutions in Place:

1. **Missing pytest plugins detection** (lines ~340-360 in daytona.py)

   - Detects "Missing required plugins:" errors
   - Automatically installs plugins like pytest-cov, pytest-xdist

2. **Recursive optional dependencies** (lines ~395-430 in daytona.py)

   - Handles "ImportError while loading conftest" errors
   - Recursively installs suggested packages (e.g., package[extras])
   - Continues until tests run or retry limit reached

3. **Missing module installation** (lines ~365-385 in daytona.py)
   - Detects ModuleNotFoundError/ImportError
   - Attempts various package name transformations
   - Special cases for common mismatches (e.g., jwt → PyJWT)

#### Types of Generic Solutions to Look For:

1. **Test Discovery Issues**

   - Pattern: "ERROR: not found:" or "collected 0 items"
   - Potential solutions:
     - Different pytest invocation methods
     - Test path format conversions
     - Working directory adjustments

2. **Configuration Issues**

   - Pattern: References to pytest.ini, setup.cfg, pyproject.toml
   - Potential solutions:
     - Temporarily modify or bypass problematic configs
     - Set pytest options to override configs

3. **Environment Setup**

   - Pattern: Tests require specific environment variables or setup
   - Potential solutions:
     - Common environment variable detection and setting
     - Pre-test setup commands based on patterns

4. **Dependency Patterns**
   - Pattern: Consistent missing packages across similar projects
   - Potential solutions:
     - Package group installations
     - Framework-specific dependency detection

### 4. Testing Changes

#### Regression Testing Protocol

Before committing any changes:

1. **Test on known-working instances** (no regression):

   ```bash
   python daytona.py 0-9 --parallel --print-exceptions --use-pbar
   ```

2. **Test on previously problematic instances** that your fix targets:

   ```bash
   python daytona.py <specific_instance_index> --logging as-you-go
   ```

3. **Run a larger batch** to ensure robustness:

   ```bash
   python daytona.py 0-99 --parallel --print-exceptions --use-pbar
   ```

4. **Verify the fundamental guarantee**:
   ```bash
   # Check that successful instances have correct test behavior
   grep -A5 -B5 "Ready for agent" test_output.log | grep -c "Tests are failing as expected"
   ```
   This count should match the number of "Ready for agent" messages.

### 5. Blacklisting Guidelines

Only blacklist instances when:

1. The issue requires compiled binaries or system-level changes
2. The fix would be highly case-specific (only helps 1-2 instances)
3. The patch doesn't actually introduce the expected bug
4. **The instance runs tests but doesn't show expected fail/pass behavior**

When blacklisting, always:

1. Add a clear comment explaining why
2. Group similar issues together
3. Document the technical reason in the comment

Example:

```python
# pandas: missing compiled C extensions (pandas._libs.pandas_parser)
"pandas-dev__pandas.95280573.lm_rewrite__am6uh57m",
```

## What to Avoid

1. **Case-Specific Logic**:

   ```python
   # BAD: Don't do this
   if "pandas" in instance_id:
       do_special_pandas_thing()
   ```

2. **Hardcoded Fixes**:

   ```python
   # BAD: Too specific
   if instance_id == "specific_instance_123":
       install_special_package()
   ```

3. **Performance-Heavy Solutions**:

   - Don't add operations that run for all instances unless necessary
   - Use patterns/error detection to trigger fixes only when needed

4. **Overly Complex Solutions**:
   - If a fix requires more than 20 lines of code, consider if there's a simpler approach
   - Complex solutions are more likely to cause regressions

## Code Reduction Checklist

When implementing new fixes, ask yourself:

1. **Can this be combined with existing logic?**

   - Look for similar patterns already handled
   - Consider extending existing solutions rather than adding new ones

2. **Is this the simplest solution?**

   - Could the same result be achieved with fewer lines?
   - Are there unnecessary conditionals or loops?

3. **Does this duplicate existing functionality?**

   - Check if another part of the code handles similar cases
   - Consider refactoring to share code

4. **Can patterns be generalized further?**
   - Instead of handling specific error messages, can you handle a class of errors?
   - Can multiple specific checks be replaced with one generic check?

## Common Patterns to Watch For

1. **Framework-Specific Issues**:

   - Django projects might need DJANGO_SETTINGS_MODULE
   - Flask projects might need FLASK_APP
   - Scientific packages often need compiled extensions

2. **Test Framework Issues**:

   - Some projects use nose, unittest, or custom test runners
   - pytest might need specific plugins or configurations

3. **Path Issues**:
   - Tests specified as module paths vs file paths
   - Tests in non-standard locations
   - Relative vs absolute path problems

## Diagnostic Enhancements

When adding diagnostics (helpful for understanding failures):

1. Keep them concise
2. Only show when relevant (e.g., when no tests run)
3. Remove verbose diagnostics once the issue is understood
4. **Never add diagnostics that slow down successful instances**

## Progress Tracking

Maintain a summary of:

- Total instances tested
- Success rate by batch
- Types of failures encountered
- Generic solutions implemented
- Instances blacklisted and why
- **Regression test results** (must always be 100% pass rate)
- **Code complexity metrics** (lines added vs removed)

## Next Steps

1. Test instances 300-399
2. Look for new failure patterns
3. Implement generic solutions for any patterns affecting multiple instances
4. Update blacklist only for instances that can't be fixed generically
5. Document findings and update this guide with new patterns discovered
6. **Run full regression suite before committing**
7. **Look for code reduction opportunities in existing solutions**

## Example Workflow for New Failure Pattern

1. **Identify Pattern**:

   ```
   Multiple instances failing with "ERROR: pytest-timeout required"
   ```

2. **Check for Existing Solutions**:

   - Is this similar to the pytest plugin detection?
   - Can we extend that solution instead of adding new code?

3. **Implement Generic Solution**:

   ```python
   # In daytona.py, add to the retry loop:
   if "pytest-timeout required" in result.result:
       await sandbox.process.exec(f"{uv_cmd} pip install -q pytest-timeout")
       continue
   ```

4. **Test Solution**:

   - Test on affected instances
   - **Run full regression test on instances 0-99**
   - Verify "Tests are failing as expected" for all successes
   - Ensure the fix is generic enough

5. **Optimize if Possible**:

   - Can this be combined with existing plugin detection?
   - Is there a more general pattern to detect?

6. **Document**:
   - Add comment in code explaining the fix
   - Update this guide if it's a new pattern category
   - Note any code that was removed or simplified

Remember: The goal is a robust, generic test runner that handles the vast majority of SWE-bench instances without case-specific code, while maintaining 100% reliability on working instances.
