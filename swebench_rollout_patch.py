#!/usr/bin/env python3
"""
Example patch for fixing the OSError: [Errno 36] File name too long issue
in SWE Bench rollout.py files.

This shows the exact changes needed to fix the RewardRunHook._get_test_results method.
"""

import asyncio
import re
from typing import Tuple, List
from command_line_fix import split_tests_into_batches, estimate_command_length, get_max_command_length


class RewardRunHook:
    """Example class showing the fixed _get_test_results method."""
    
    def __init__(self, instance, trajectory, run_single):
        self.instance = instance
        self.trajectory = trajectory
        self.run_single = run_single
    
    def _get_test_results_original(self, tests: List[str]) -> Tuple[int, int]:
        """
        ORIGINAL (PROBLEMATIC) VERSION - DON'T USE THIS
        
        This is the version that causes OSError: [Errno 36] File name too long
        """
        observation = asyncio.run(
            self.run_single.env.deployment.runtime.run_in_session(
                BashAction(
                    command=f"cd /testbed && python -m pytest {' '.join(tests)}",
                    check="silent",
                    timeout=1200.0,
                )
            )
        )
        summary_line = observation.output.splitlines()[-1]
        failed_match = re.search(r"(\d+)\s+failed", summary_line)
        passed_match = re.search(r"(\d+)\s+passed", summary_line)
        
        num_failed = int(failed_match.group(1)) if failed_match else 0
        num_passed = int(passed_match.group(1)) if passed_match else 0
        
        return num_failed, num_passed
    
    async def _get_test_results_fixed(self, tests: List[str]) -> Tuple[int, int]:
        """
        FIXED VERSION - USE THIS
        
        This version handles long test lists by batching them when necessary.
        """
        if not tests:
            return 0, 0
        
        base_cmd = "cd /testbed && python -m pytest"
        max_cmd_length = get_max_command_length()
        
        # Check if we need to batch the tests
        total_length = estimate_command_length(base_cmd, tests)
        
        if total_length <= max_cmd_length:
            # Small list - use original approach
            return await self._run_single_pytest_command(tests)
        else:
            # Large list - use batching approach
            return await self._run_batched_pytest_commands(tests)
    
    async def _run_single_pytest_command(self, tests: List[str]) -> Tuple[int, int]:
        """Run all tests in a single pytest command."""
        from swerex.runtime.abstract import BashAction
        
        try:
            observation = await self.run_single.env.deployment.runtime.run_in_session(
                BashAction(
                    command=f"cd /testbed && python -m pytest {' '.join(tests)}",
                    check="silent",
                    timeout=1200.0,
                )
            )
            
            return self._parse_pytest_output(observation.output)
            
        except OSError as e:
            if "File name too long" in str(e):
                # Fall back to batching if we somehow still hit the limit
                return await self._run_batched_pytest_commands(tests)
            else:
                raise
    
    async def _run_batched_pytest_commands(self, tests: List[str]) -> Tuple[int, int]:
        """Run tests in multiple batches to avoid command line length issues."""
        from swerex.runtime.abstract import BashAction
        
        base_cmd = "cd /testbed && python -m pytest"
        batches = split_tests_into_batches(tests, base_cmd)
        
        total_failed = 0
        total_passed = 0
        
        print(f"Running {len(tests)} tests in {len(batches)} batches to avoid command line length issues")
        
        for i, batch in enumerate(batches):
            print(f"Running batch {i+1}/{len(batches)} with {len(batch)} tests...")
            
            try:
                observation = await self.run_single.env.deployment.runtime.run_in_session(
                    BashAction(
                        command=f"cd /testbed && python -m pytest {' '.join(batch)}",
                        check="silent",
                        timeout=1200.0,
                    )
                )
                
                batch_failed, batch_passed = self._parse_pytest_output(observation.output)
                total_failed += batch_failed
                total_passed += batch_passed
                
                print(f"Batch {i+1} results: {batch_passed} passed, {batch_failed} failed")
                
            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                # Count all tests in this batch as failed
                total_failed += len(batch)
        
        print(f"Total results: {total_passed} passed, {total_failed} failed")
        return total_failed, total_passed
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int]:
        """Parse pytest output to extract pass/fail counts."""
        if not output:
            return 0, 0
        
        lines = output.splitlines()
        if not lines:
            return 0, 0
        
        summary_line = lines[-1]
        failed_match = re.search(r"(\d+)\s+failed", summary_line)
        passed_match = re.search(r"(\d+)\s+passed", summary_line)
        
        num_failed = int(failed_match.group(1)) if failed_match else 0
        num_passed = int(passed_match.group(1)) if passed_match else 0
        
        return num_failed, num_passed
    
    def on_instance_completed(self, *, result) -> None:
        """
        FIXED VERSION of the method that calls _get_test_results
        """
        # Get test results for FAIL_TO_PASS tests
        num_failed_f2p, num_passed_f2p = asyncio.run(
            self._get_test_results_fixed(self.instance["FAIL_TO_PASS"])
        )
        
        # Get test results for PASS_TO_PASS tests  
        num_failed_p2p, num_passed_p2p = asyncio.run(
            self._get_test_results_fixed(self.instance["PASS_TO_PASS"])
        )
        
        # Update trajectory with results
        update_trajectory(
            self.trajectory,
            self.instance,
            result,
            num_failed_f2p,
            num_passed_f2p,
            num_failed_p2p,
            num_passed_p2p,
        )


# Example of how to apply the patch:

def apply_patch_example():
    """
    Example showing exactly what to change in your rollout.py file.
    """
    print("""
    TO FIX THE ISSUE:
    
    1. Add this import at the top of your rollout.py file:
       from command_line_fix import split_tests_into_batches, estimate_command_length, get_max_command_length
    
    2. Replace the _get_test_results method in RewardRunHook class with the fixed version above.
    
    3. Change the synchronous call to async:
       
       BEFORE:
       num_failed_f2p, num_passed_f2p = self._get_test_results(
           self.instance["FAIL_TO_PASS"]
       )
       
       AFTER:
       num_failed_f2p, num_passed_f2p = await self._get_test_results_fixed(
           self.instance["FAIL_TO_PASS"]
       )
    
    4. Make sure the calling method is async and uses await.
    
    That's it! The error should be resolved.
    """)


if __name__ == "__main__":
    apply_patch_example()
    
    # Example usage
    print("\nExample of how the fixed method handles large test lists:")
    
    # Simulate a large test list like what might cause the error
    large_test_list = [
        f"tests/test_module_{i}/test_file_{j}.py::TestClass::test_method_{k}"
        for i in range(10)
        for j in range(10)
        for k in range(5)
    ]
    
    print(f"Example with {len(large_test_list)} tests:")
    
    base_cmd = "cd /testbed && python -m pytest"
    total_length = estimate_command_length(base_cmd, large_test_list)
    max_length = get_max_command_length()
    
    print(f"  Command length: {total_length} characters")
    print(f"  System limit: {max_length} characters")
    
    if total_length > max_length:
        batches = split_tests_into_batches(large_test_list, base_cmd)
        print(f"  Would be split into {len(batches)} batches")
    else:
        print("  Would run as single command")