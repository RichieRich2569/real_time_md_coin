function rec = captureRun(scenario)
%CAPTURERUN Replay one scenario and record the per-trial observable surface.
%
%   rec = equiv.captureRun(scenario) constructs a RealTimeCOIN from
%   scenario.args, seeds the model RNG with scenario.seed, streams the fixed
%   (q, y) inputs one trial at a time, and records a set of query-method
%   outputs after every trial. The recording is the comparison surface used by
%   equiv.compareRuns: if any bit of the internal particle state diverges, at
%   least one of these observables changes.
%
%   rec is a struct whose fields are 1-by-T cell arrays of numeric column
%   vectors (one cell per trial). Fields:
%     motor            - motor_output (predictive feedback mean, dim-by-1)
%     predCtxProb      - predicted_context_probabilities (row -> column)
%     resp             - responsibilities (row -> column)
%     sampledCount     - sampled_context_count (scalar)
%     stateMean        - state_moments first output (posterior state mean)
%
%   Uses whichever RealTimeCOIN is first on the MATLAB path, so the caller
%   controls reference (main) vs candidate (experimental) via path order.
%
%   See also equiv.scenarioBattery, equiv.captureAll, equiv.compareRuns.

    rng(scenario.seed, 'twister');
    obj = RealTimeCOIN(scenario.args{:});

    T = size(scenario.y, 2);
    fields = {'motor', 'predCtxProb', 'resp', 'sampledCount', 'stateMean'};
    for f = 1:numel(fields)
        rec.(fields{f}) = cell(1, T);
    end

    % Old query names still resolve after the Batch 3 rename via deprecation
    % shims; silence those warnings so the capture output stays clean.
    ws = warning('off', 'all');
    cleanup = onCleanup(@() warning(ws));

    for t = 1:T
        obj.observe_q(scenario.q(t));
        obj.observe_y(scenario.y(:, t));

        rec.motor{t}        = columnize(obj.motor_output());
        rec.predCtxProb{t}  = columnize(obj.predicted_context_probabilities());
        rec.resp{t}         = columnize(obj.responsibilities());
        rec.sampledCount{t} = columnize(obj.sampled_context_count());
        rec.stateMean{t}    = columnize(firstOutput(@() obj.state_moments()));
    end
end

function v = columnize(x)
%COLUMNIZE Flatten any numeric to a double column vector (NaN-preserving).
    if isnumeric(x) || islogical(x)
        v = double(x(:));
    else
        v = NaN;   % non-numeric (e.g. a Map) is not part of this surface
    end
end

function out = firstOutput(fn)
%FIRSTOUTPUT Return the first output of fn, or NaN if it errors.
    try
        out = fn();
    catch
        out = NaN;
    end
end
