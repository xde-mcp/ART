#!/usr/bin/env python3
"""
Solution for OSError: [Errno 36] File name too long when running pytest with many test files.

This module provides utilities to run pytest commands with large numbers of test files
by splitting them into smaller batches to avoid exceeding shell command line length limits.
"""

import re
import subprocess
import sys
from typing import List, Tuple, Union
import shlex
import os


def get_max_command_length() -> int:
    """
    Get the maximum command line length for the current system.
    
    Returns:
        Maximum command line length in characters
    """
    try:
        # Try to get the actual system limit
        result = subprocess.run(['getconf', 'ARG_MAX'], capture_output=True, text=True)
        if result.returncode == 0:
            # ARG_MAX includes environment variables, so we use a conservative estimate
            arg_max = int(result.stdout.strip())
            # Reserve space for environment variables and other arguments
            return min(arg_max // 2, 100000)  # Conservative limit
        else:
            # Fallback to a safe default
            return 32000
    except Exception:
        # Ultra-conservative fallback
        return 32000


def estimate_command_length(base_cmd: str, tests: List[str]) -> int:
    """
    Estimate the total command length including all test arguments.
    
    Args:
        base_cmd: Base command (e.g., "cd /testbed && python -m pytest")
        tests: List of test files/patterns
        
    Returns:
        Estimated command length in characters
    """
    # Account for spaces between arguments and shell escaping
    test_args = ' '.join(shlex.quote(test) for test in tests)
    return len(base_cmd) + len(test_args) + 10  # 10 chars buffer


def split_tests_into_batches(tests: List[str], base_cmd: str, max_length: int = None) -> List[List[str]]:
    """
    Split a list of tests into batches that won't exceed command line length limits.
    
    Args:
        tests: List of test files/patterns
        base_cmd: Base command that will be used
        max_length: Maximum command length (if None, will be auto-detected)
        
    Returns:
        List of test batches, where each batch can be safely used in a command
    """
    if max_length is None:
        max_length = get_max_command_length()
    
    if not tests:
        return []
    
    # If the command with all tests is within limits, return as single batch
    if estimate_command_length(base_cmd, tests) <= max_length:
        return [tests]
    
    batches = []
    current_batch = []
    
    for test in tests:
        # Create a test batch with the current test added
        test_batch = current_batch + [test]
        
        # Check if this batch would exceed the limit
        if estimate_command_length(base_cmd, test_batch) > max_length:
            # If current_batch is empty, this single test is too long
            if not current_batch:
                # Split this single test if it's a complex pattern
                # For now, just add it and hope for the best
                batches.append([test])
            else:
                # Add the current batch and start a new one
                batches.append(current_batch)
                current_batch = [test]
        else:
            # This batch is still within limits
            current_batch = test_batch
    
    # Add the final batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


def run_pytest_in_batches(tests: List[str], base_dir: str = "/testbed", 
                         extra_args: List[str] = None, timeout: float = 1200.0) -> Tuple[int, int]:
    """
    Run pytest with a list of tests, automatically batching if necessary to avoid command line length issues.
    
    Args:
        tests: List of test files/patterns to run
        base_dir: Directory to run tests from
        extra_args: Additional arguments to pass to pytest
        timeout: Timeout for each batch in seconds
        
    Returns:
        Tuple of (total_failed, total_passed) across all batches
    """
    if extra_args is None:
        extra_args = []
    
    base_cmd = f"cd {shlex.quote(base_dir)} && python -m pytest"
    batches = split_tests_into_batches(tests, base_cmd)
    
    total_failed = 0
    total_passed = 0
    
    for i, batch in enumerate(batches):
        print(f"Running test batch {i+1}/{len(batches)} with {len(batch)} tests...")
        
        # Construct the full command
        cmd_parts = ["cd", base_dir, "&&", "python", "-m", "pytest"] + extra_args + batch
        cmd = ' '.join(shlex.quote(part) for part in cmd_parts)
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse pytest output to get results
            output_lines = result.stdout.splitlines()
            if output_lines:
                summary_line = output_lines[-1]
                failed_match = re.search(r'(\d+)\s+failed', summary_line)
                passed_match = re.search(r'(\d+)\s+passed', summary_line)
                
                batch_failed = int(failed_match.group(1)) if failed_match else 0
                batch_passed = int(passed_match.group(1)) if passed_match else 0
                
                total_failed += batch_failed
                total_passed += batch_passed
                
                print(f"Batch {i+1} results: {batch_passed} passed, {batch_failed} failed")
            
        except subprocess.TimeoutExpired:
            print(f"Batch {i+1} timed out after {timeout} seconds")
            # Count all tests in this batch as failed
            total_failed += len(batch)
        except Exception as e:
            print(f"Error running batch {i+1}: {e}")
            # Count all tests in this batch as failed
            total_failed += len(batch)
    
    print(f"Total results: {total_passed} passed, {total_failed} failed")
    return total_failed, total_passed


# Example usage function that mimics the pattern from the error trace
async def run_tests_safe(tests: List[str], runtime_env) -> Tuple[int, int]:
    """
    Safe version of the test running function that caused the original error.
    
    This function can be used as a drop-in replacement for the problematic
    _get_test_results method pattern.
    
    Args:
        tests: List of test files/patterns
        runtime_env: Runtime environment (for compatibility)
        
    Returns:
        Tuple of (num_failed, num_passed)
    """
    # If the test list is small, try the original approach first
    if len(tests) <= 10:
        test_string = ' '.join(tests)
        if len(test_string) < 1000:  # Conservative threshold
            try:
                # Try the original command format
                from swerex.runtime.abstract import BashAction
                
                observation = await runtime_env.deployment.runtime.run_in_session(
                    BashAction(
                        command=f"cd /testbed && python -m pytest {test_string}",
                        check="silent",
                        timeout=1200.0,
                    )
                )
                
                # Parse the results
                summary_line = observation.output.splitlines()[-1]
                failed_match = re.search(r"(\d+)\s+failed", summary_line)
                passed_match = re.search(r"(\d+)\s+passed", summary_line)
                
                num_failed = int(failed_match.group(1)) if failed_match else 0
                num_passed = int(passed_match.group(1)) if passed_match else 0
                
                return num_failed, num_passed
                
            except OSError as e:
                if "File name too long" not in str(e):
                    raise  # Re-raise if it's a different error
                # Fall through to batched approach
    
    # Use the batched approach
    return run_pytest_in_batches(tests)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pytest with large test lists safely")
    parser.add_argument("tests", nargs="+", help="Test files or patterns to run")
    parser.add_argument("--base-dir", default="/testbed", help="Base directory for tests")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout per batch")
    
    args = parser.parse_args()
    
    num_failed, num_passed = run_pytest_in_batches(
        args.tests, 
        base_dir=args.base_dir,
        timeout=args.timeout
    )
    
    print(f"\nFinal results: {num_passed} passed, {num_failed} failed")
    sys.exit(1 if num_failed > 0 else 0)