function assertClose(actual, expected, tol, msg)
%TESTUTIL.ASSERTCLOSE Assert two arrays match elementwise within a tolerance.
%   testutil.assertClose(actual, expected, tol, msg) fails when the largest
%   absolute difference between actual and expected exceeds tol. The failure
%   message appends the observed maximum absolute error.
err = max(abs(actual(:) - expected(:)));
assert(err <= tol, sprintf('%s (max abs err %.3g)', msg, err));
end
