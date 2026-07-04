function p = predictive_cue_p_value(obj, q, u)
    %PREDICTIVE_CUE_P_VALUE Predictive cue p-value for a given cue label.
    %   p = predictive_cue_p_value(obj, q, u) returns the predictive p-value for a
    %   given cue label q, using a uniform random variable u in [0,1]. If q is empty, returns NaN. If q is a cue label that has not been observed, returns 0. If u is not provided, it defaults to a uniform random variable in [0,1]. 
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty, mustBeInteger, mustBeFinite, mustBeNonnegative} = [];
        u (1, 1) double {mustBeFinite} = rand;
    end
    qLabel = peekCueLabel(obj, q);
    if isempty(qLabel)
        p = NaN;
        return;
    end
    [pmf, labels] = previewCuePmf(obj);
    if qLabel > numel(pmf)
        pmf(end+1:qLabel) = 0;
        labels = 1:numel(pmf);
    end
    f = pmf(qLabel);
    Fminus = sum(pmf(labels < qLabel));
    p = min(max(Fminus + u .* f, 0), 1);
end

function mustBeScalarOrEmpty(x)
    if ~(isempty(x) || isscalar(x))
        error('RealTimeCOIN:InvalidCue', ...
            'q must be empty or a scalar cue label.');
    end
end