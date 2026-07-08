function assertSize(name, v, expected)
%TESTUTIL.ASSERTSIZE Assert size(v) equals the expected size vector.
%   testutil.assertSize(name, v, expected) errors when size(v) differs from
%   expected, reporting both sizes alongside name.
    if ~isequal(size(v), expected)
        error('testutil:assertSize:failed', 'FAILED: %s size [%s] != [%s]', ...
            name, num2str(size(v)), num2str(expected));
    end
end
