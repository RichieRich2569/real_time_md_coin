function report = compareRuns(refCaps, testCaps)
%COMPARERUNS Diff two equivalence-battery captures (reference vs candidate).
%
%   report = equiv.compareRuns(refCaps, testCaps) compares the per-scenario
%   recordings produced by equiv.captureAll on two code versions. For every
%   scenario/field/trial it computes the maximum absolute difference, treating
%   NaN-in-both as a match and any NaN-pattern mismatch or size mismatch as a
%   structural failure (Inf difference).
%
%   report is a struct with:
%     names        - 1-by-N scenario names
%     maxAbsDiff   - 1-by-N max abs difference per scenario
%     structMismatch - 1-by-N logical, true if sizes/NaN-patterns disagreed
%     classA       - 1-by-N logical, true when maxAbsDiff == 0 and no mismatch
%     overallMax   - scalar max over all scenarios
%     allClassA    - true when every scenario is bit-identical
%
%   A scenario failing classA is not necessarily a bug: for trajectory-changing
%   (Class B) optimizations, use the ensemble gate (run_validation vs COIN.m and
%   motor-output RMSE vs main) instead. classA is the bit-identical detector.
%
%   See also equiv.captureAll, equiv.captureRun.

    scen = equiv.scenarioBattery();
    n = numel(refCaps);
    report.names = {scen.name};
    report.maxAbsDiff = zeros(1, n);
    report.structMismatch = false(1, n);

    for i = 1:n
        [d, mism] = diffRecording(refCaps{i}, testCaps{i});
        report.maxAbsDiff(i) = d;
        report.structMismatch(i) = mism;
    end

    report.classA = (report.maxAbsDiff == 0) & ~report.structMismatch;
    report.overallMax = max(report.maxAbsDiff);
    report.allClassA = all(report.classA);

    fprintf('\nEquivalence report (%d scenarios):\n', n);
    for i = 1:n
        tag = 'CLASS-A (bit-identical)';
        if report.structMismatch(i)
            tag = 'STRUCT-MISMATCH';
        elseif report.maxAbsDiff(i) > 0
            tag = 'DIVERGED (Class B candidate)';
        end
        fprintf('  %-20s maxAbsDiff=%.3e  %s\n', ...
            report.names{i}, report.maxAbsDiff(i), tag);
    end
    fprintf('  overall max abs diff = %.3e ; all Class A = %d\n', ...
        report.overallMax, report.allClassA);
end

function [maxDiff, mismatch] = diffRecording(a, b)
%DIFFRECORDING Max abs diff across all fields/trials of two recordings.
    maxDiff = 0;
    mismatch = false;
    fields = fieldnames(a);
    for f = 1:numel(fields)
        fld = fields{f};
        if ~isfield(b, fld)
            mismatch = true; maxDiff = Inf; return;
        end
        ca = a.(fld); cb = b.(fld);
        if numel(ca) ~= numel(cb)
            mismatch = true; maxDiff = Inf; return;
        end
        for t = 1:numel(ca)
            va = ca{t}; vb = cb{t};
            if ~isequal(size(va), size(vb))
                mismatch = true; maxDiff = Inf; return;
            end
            nanA = isnan(va); nanB = isnan(vb);
            if ~isequal(nanA, nanB)
                mismatch = true; maxDiff = Inf; return;
            end
            good = ~nanA;
            if any(good)
                d = max(abs(va(good) - vb(good)));
                maxDiff = max(maxDiff, d);
            end
        end
    end
end
