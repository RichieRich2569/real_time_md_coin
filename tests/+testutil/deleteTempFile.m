function deleteTempFile(path)
%TESTUTIL.DELETETEMPFILE Delete a temporary file if it exists.
%   testutil.deleteTempFile(path) removes the file at path when present and is a
%   no-op otherwise. Intended for use with onCleanup so temporary files are
%   removed even when a test errors, without warning if the file was never
%   created.
    if (ischar(path) || (isstring(path) && isscalar(path))) && isfile(path)
        delete(path);
    end
end
