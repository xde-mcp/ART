# Findings from Testing Instances 300-399

## Summary
- **Total instances tested**: 100 (instances 300-399)
- **Successful**: 93 instances (93%)
- **Failed (no tests executed)**: 5 instances (5%)
- **Other failures**: 2 instances (2%)

## Instances with No Tests Executed
1. `django-money__django-money.835c1ab8.lm_rewrite__bv3wb2wq` - NEW issue
2. `kayak__pypika.1c9646f0.lm_rewrite__xdfkt9wb` - Known PyPika issue
3. `kayak__pypika.1c9646f0.func_pm_class_rm_funcs__muwfrdsf` - Known PyPika issue  
4. `pandas-dev__pandas.95280573.pr_48966` - Known Pandas issue
5. `pandas-dev__pandas.95280573.pr_48426` - Known Pandas issue

## New Patterns Identified
- **Django-money instances**: Similar to PyPika, these instances may have test collection issues
- No new generic solution patterns were identified - all failures match existing blacklist patterns

## Recommendations
1. Add the 5 failed instances to the blacklist
2. Continue with regression testing on instances 0-99 to ensure no regressions
3. The 93% success rate is consistent with the overall 94% success rate

## UV Migration
Successfully using `uv` for package management, which provides:
- Faster package installation
- Better dependency resolution
- More reliable package management