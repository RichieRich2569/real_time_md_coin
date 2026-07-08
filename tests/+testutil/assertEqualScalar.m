function assertEqualScalar(name, a, b, tol)
%TESTUTIL.ASSERTEQUALSCALAR Assert two values match within a tolerance.
%   testutil.assertEqualScalar(name, a, b, tol) errors when any elementwise
%   absolute difference between a and b exceeds tol, reporting the maximum
%   difference alongside name.
    if any(abs(a(:) - b(:)) > tol)
        error('testutil:assertEqualScalar:failed', 'FAILED: %s (max diff %.3g)', name, max(abs(a(:)-b(:))));
    end
end
