%RUN_TESTS Execute all test scripts in this directory.
%
%   This helper script runs each test function in the tests folder.

fprintf('Running RealTimeCOIN tests...\n');
files = dir(fullfile(fileparts(mfilename('fullpath')), 'test_*.m'));
for k = 1:length(files)
    testName = files(k).name(1:end-2);
    fprintf('Running %s...\n', testName);
    feval(testName);
end
fprintf('All tests completed.\n');