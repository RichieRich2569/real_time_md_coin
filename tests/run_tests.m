function run_tests
%RUN_TESTS Execute every test_*.m in this folder and report a pass/fail tally.
%
%   Each test function runs inside its own try/catch so a single failure does not
%   halt the suite; the failing test's error message is printed and the run
%   continues. A summary line reports how many tests passed and names any that
%   failed. Tests remain plain functions (not matlab.unittest).

testDir = fileparts(mfilename('fullpath'));
files = dir(fullfile(testDir, 'test_*.m'));
names = {files.name};

fprintf('Running RealTimeCOIN tests...\n');
numPassed = 0;
failures = {};
for k = 1:numel(names)
    testName = names{k}(1:end-2);
    fprintf('Running %s...\n', testName);
    try
        feval(testName);
        numPassed = numPassed + 1;
    catch err
        failures{end+1} = testName; %#ok<AGROW>
        fprintf(2, '  FAILED: %s\n', err.message);
    end
end

fprintf('\n%d/%d tests passed.\n', numPassed, numel(names));
if ~isempty(failures)
    fprintf(2, 'Failed tests: %s\n', strjoin(failures, ', '));
end
end
