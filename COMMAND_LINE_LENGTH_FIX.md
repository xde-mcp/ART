# Fix for OSError: [Errno 36] File name too long

## Problem Description

When running SWE Bench or similar test suites, you may encounter this error:

```
OSError: [Errno 36] File name too long: '/bin/sh'
```

This error is misleading - it's not actually that `/bin/sh` is too long, but rather that the command being passed to the shell exceeds the system's maximum command line length limit.

### What causes this?

The error typically occurs when running pytest with many test files:

```python
BashAction(
    command=f"cd /testbed && python -m pytest {' '.join(tests)}",
    check="silent",
    timeout=1200.0,
)
```

When `tests` contains many test files (common in SWE Bench instances), `' '.join(tests)` creates a command line that can exceed:
- Linux ARG_MAX limit (typically 128KB, but varies)
- Shell command length limits
- Environment variable space limits

### Example problematic scenario:

```python
tests = [
    "test_file_1.py::test_function_1", 
    "test_file_2.py::test_function_2",
    # ... hundreds more test patterns
]
# This creates a command like:
# cd /testbed && python -m pytest test_file_1.py::test_function_1 test_file_2.py::test_function_2 ... (very long)
```

## Solution

The `command_line_fix.py` module provides utilities to automatically batch large test lists to avoid command line length issues.

### Quick Fix

Replace problematic test execution code:

```python
# BEFORE (problematic):
def _get_test_results(self, tests: list[str]) -> tuple[int, int]:
    observation = asyncio.run(
        self.run_single.env.deployment.runtime.run_in_session(
            BashAction(
                command=f"cd /testbed && python -m pytest {' '.join(tests)}",
                check="silent",
                timeout=1200.0,
            )
        )
    )
    # ... parse results

# AFTER (fixed):
from command_line_fix import run_tests_safe

async def _get_test_results(self, tests: list[str]) -> tuple[int, int]:
    return await run_tests_safe(tests, self.run_single.env)
```

### Manual batching approach

```python
from command_line_fix import split_tests_into_batches, run_pytest_in_batches

# Split tests into safe batches
tests = ["test1.py", "test2.py", ...]  # Your long list of tests
base_cmd = "cd /testbed && python -m pytest"
batches = split_tests_into_batches(tests, base_cmd)

print(f"Split {len(tests)} tests into {len(batches)} batches")

# Run all batches
total_failed, total_passed = run_pytest_in_batches(tests)
```

### Command line usage

```bash
python command_line_fix.py test1.py test2.py test3.py ... testN.py
```

## Key Features

1. **Automatic detection**: Detects system command line limits
2. **Smart batching**: Only splits when necessary
3. **Conservative limits**: Uses safe defaults to avoid edge cases
4. **Result aggregation**: Combines results from all batches
5. **Error handling**: Gracefully handles timeouts and failures
6. **Drop-in replacement**: `run_tests_safe()` can replace existing problematic code

## How it works

1. **Estimate command length**: Calculates total command length including shell escaping
2. **Detect system limits**: Uses `getconf ARG_MAX` or safe defaults
3. **Batch intelligently**: Splits tests into the minimum number of batches needed
4. **Aggregate results**: Combines pytest results from all batches

## System Limits Reference

| System | Typical ARG_MAX | Conservative Limit Used |
|--------|----------------|-------------------------|
| Linux  | 128KB - 2MB    | 32KB - 100KB           |
| macOS  | 256KB          | 32KB - 100KB           |
| Windows| 8KB            | 32KB                   |

## Integration with SWE Bench

For SWE Bench training pipelines, you can modify the `RewardRunHook._get_test_results` method:

```python
# In your rollout.py or similar file:
from command_line_fix import run_tests_safe

class RewardRunHook:
    async def _get_test_results(self, tests: list[str]) -> tuple[int, int]:
        """Safe version that handles long test lists."""
        return await run_tests_safe(tests, self.run_single.env)
```

This will automatically:
- Try the original single command approach for small test lists
- Fall back to batching for large test lists
- Handle the `OSError: [Errno 36]` gracefully
- Return the same `(num_failed, num_passed)` tuple format

## Testing the fix

```python
# Test with a large number of dummy tests
large_test_list = [f"test_file_{i}.py::test_function_{i}" for i in range(1000)]

# This would fail with the original approach
# But works fine with the batched approach
num_failed, num_passed = run_pytest_in_batches(large_test_list)
```

## Additional considerations

- **Performance**: Batching adds some overhead but is necessary for correctness
- **Parallelization**: Consider running batches in parallel if your system supports it
- **Temporary files**: Alternative approach using pytest's `--collect-only` and test files
- **Configuration**: Adjust batch sizes based on your specific system and test characteristics