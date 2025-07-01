# Test Run Summary: Instances 300-399

## Summary Statistics
- **Total instances tested**: 69 (some may have been filtered out from the 100 range)
- **Successful**: 62 (89.9%)
- **Failed**: 7 (10.1%)
- **Blacklisted**: 2

## Failed Instances

### PyPika Instances (2 failures)
- `kayak__pypika.1c9646f0.lm_rewrite__xdfkt9wb` - No tests executed
- `kayak__pypika.1c9646f0.func_pm_class_rm_funcs__muwfrdsf` - No tests executed

### Pandas Instances (3 failures)
- `pandas-dev__pandas.95280573.pr_48426` - No tests executed
- `pandas-dev__pandas.95280573.pr_48966` - No tests executed
- One more pandas instance (from already blacklisted)

### Django-Money Instance (1 failure)
- `django-money__django-money.835c1ab8.lm_rewrite__bv3wb2wq` - No tests executed

### Timeout Errors (2 failures)
- Two instances resulted in TimeoutError (specific instances not identified in log)

## Analysis

### Pattern 1: PyPika Test Collection Issues
Both PyPika instances failed with "No tests executed". This is consistent with the pattern seen in instances 0-299, where PyPika instances had test collection issues.

### Pattern 2: Pandas Compilation Issues
The pandas instances likely failed due to missing compiled C extensions (`pandas._libs.pandas_parser`), which is also a known issue from the previous test runs.

### Pattern 3: Django-Money
This appears to be a new pattern not seen in instances 0-299.

### Pattern 4: Timeout Errors
These may be due to tests taking too long or getting stuck in infinite loops.

## Recommendations

1. **PyPika instances**: Should be added to the blacklist as they consistently fail with test collection issues
2. **Pandas instances**: Already have a pattern in the blacklist for pandas compilation issues
3. **Django-Money**: Needs investigation to determine if this is a generic issue
4. **Timeout errors**: May need to identify specific instances and determine if timeout limits need adjustment