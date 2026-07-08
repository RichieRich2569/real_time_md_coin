function assertInRange(name, v, lo, hi)
%TESTUTIL.ASSERTINRANGE Assert every element of v lies in [lo, hi].
%   testutil.assertInRange(name, v, lo, hi) errors when any element of v falls
%   below lo or above hi (each bound relaxed by 1e-12), reporting name.
    if any(v(:) < lo - 1e-12) || any(v(:) > hi + 1e-12)
        error('testutil:assertInRange:failed', 'FAILED: %s = %.6g not in [%g,%g]', name, v, lo, hi);
    end
end
