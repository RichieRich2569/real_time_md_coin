function mustError(name, fn, expectedId)
%TESTUTIL.MUSTERROR Assert a function handle throws a specific error identifier.
%   testutil.mustError(name, fn, expectedId) calls fn() and requires it to
%   raise an error whose identifier equals expectedId. If fn does not error, or
%   errors with a different identifier, the original (or a synthesised "did not
%   error") failure propagates.
    try
        fn();
        error('testutil:mustError:noError', 'FAILED: %s did not error', name);
    catch e
        if ~strcmp(e.identifier, expectedId)
            rethrow(e);
        end
    end
end
