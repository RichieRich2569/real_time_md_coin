function caps = captureAll()
%CAPTUREALL Run the full equivalence battery against the RealTimeCOIN on path.
%
%   caps = equiv.captureAll() returns a 1-by-N cell array of per-scenario
%   recordings (see equiv.captureRun) for every scenario in
%   equiv.scenarioBattery. It does NOT manage the MATLAB path: the caller is
%   responsible for putting the desired @RealTimeCOIN root first on the path
%   (and issuing "clear classes; rehash") before calling this, so the same
%   function captures either the reference (main) or the candidate
%   (experimental) code.
%
%   See also equiv.captureRun, equiv.compareRuns, equiv.scenarioBattery.

    scen = equiv.scenarioBattery();
    caps = cell(1, numel(scen));
    for i = 1:numel(scen)
        caps{i} = equiv.captureRun(scen(i));
    end
end
