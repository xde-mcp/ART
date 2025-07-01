# Instructions for Continuing SWE-Bench Daytona Test Runner Evaluation

## Current Status
Successfully completed testing instances 0-399 with significant improvements implemented. The system is now much more robust and handles various failure patterns effectively.

## Latest Improvements (Just Implemented)
1. **Enhanced Pandas Installation**: Proper compilation sequence for pandas._libs.pandas_parser
2. **Network Timeout Handling**: Exponential backoff retry for 504 Gateway errors  
3. **Expanded Dependency Detection**: Better module-to-package mappings
4. **Improved Error Diagnostics**: Clear failure analysis and graceful degradation

## Next Steps

### Immediate Tasks (Priority 1)
1. **Continue systematic testing**:
   ```bash
   python3 daytona.py 400-499 --parallel --print-exceptions --use-pbar
   ```

2. **Analyze new failure patterns** from instances 400-499:
   - Look for any new dependency issues
   - Identify any new types of test collection failures
   - Note any performance issues or timeouts

3. **Update blacklist** if needed:
   - Add instances that consistently fail to execute tests
   - Focus on cases where "No tests were executed" occurs repeatedly

### Medium-term Improvements (Priority 2)
1. **Enhance test collection debugging**:
   - Add more diagnostic information for pytest collection failures
   - Implement fallback strategies for problematic test paths
   - Consider alternative test runners for specific frameworks

2. **Performance optimizations**:
   - Fine-tune timeout values based on observed patterns
   - Optimize parallel execution batching
   - Implement smarter retry strategies

3. **Advanced dependency handling**:
   - Add support for conda packages where pip fails
   - Handle C extension compilation issues better
   - Implement package-specific installation strategies

### Implementation Guidelines

#### When Adding New Improvements
1. **Test on small batches first** (e.g., 10-20 instances) before running large batches
2. **Document new patterns** in CURRENT_STATUS.md
3. **Add comprehensive error handling** with graceful fallbacks
4. **Maintain backward compatibility** with existing functionality

#### Code Quality Standards
- Add type hints for new functions
- Include comprehensive docstrings
- Use meaningful variable names
- Handle exceptions gracefully
- Add logging for debugging

#### Testing Strategy
```bash
# Test small batch first
python3 daytona.py 400-409 --parallel --print-exceptions --use-pbar

# If successful, test larger batch
python3 daytona.py 400-449 --parallel --print-exceptions --use-pbar

# Finally test full range
python3 daytona.py 400-499 --parallel --print-exceptions --use-pbar
```

## Known Issues to Monitor

### Persistent Problems
1. **Pandas compilation**: Some pandas instances may still fail despite improvements
2. **Complex build requirements**: Packages with native dependencies
3. **Test path issues**: Some instances have incorrect test file references
4. **Configuration conflicts**: Pytest configuration incompatibilities

### Infrastructure Issues
1. **Network timeouts**: Though improved, may still occur under load
2. **Sandbox resource limits**: Memory/CPU constraints for large test suites
3. **Dependency conflicts**: Version conflicts between packages

## Success Criteria for Each Batch
- **>90% instances processed**: Either successful test execution or clear failure diagnosis
- **<5% timeout failures**: Network issues should be rare with retry logic
- **Clear failure categorization**: All failures should be properly diagnosed and logged
- **Performance targets**: <5 seconds average per instance, <30 instances per minute throughput

## Reporting Guidelines

### After Each Batch (e.g., 400-499)
1. **Update CURRENT_STATUS.md** with:
   - New completion range
   - Success metrics (% processed, failure types)
   - Any new patterns discovered
   - Performance observations

2. **Document new improvements** in code with:
   - Clear problem description
   - Solution implemented
   - Testing validation

3. **Update blacklist candidates** if instances consistently fail

### When Implementation is Complete
1. **Comprehensive summary report** of all improvements
2. **Performance benchmarks** and optimization recommendations
3. **Maintenance guidelines** for ongoing operation
4. **Documentation** for deploying the improved system

## Files to Monitor
- `daytona.py` - Main test runner implementation
- `instances.py` - Instance management and filtering
- `CURRENT_STATUS.md` - Status tracking and improvements log
- `test_run_*.log` - Detailed test execution logs

## Emergency Procedures
If systematic failures occur:
1. **Stop parallel execution** and test single instances
2. **Check sandbox availability** and resource limits
3. **Verify network connectivity** to Daytona services
4. **Rollback recent changes** if they caused regressions
5. **Document the issue** and recovery steps taken

The system is in good shape and ready for continued systematic evaluation. Focus on maintaining quality while progressing through the remaining instances.