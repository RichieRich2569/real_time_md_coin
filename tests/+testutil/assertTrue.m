function assertTrue(name, cond)
%TESTUTIL.ASSERTTRUE Assert a named boolean condition holds.
%   testutil.assertTrue(name, cond) errors, reporting name, when cond is false.
    if ~cond
        error('testutil:assertTrue:failed', 'FAILED: %s', name);
    end
end
