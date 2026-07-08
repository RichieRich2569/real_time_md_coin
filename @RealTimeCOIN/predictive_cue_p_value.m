function p = predictive_cue_p_value(obj, q, u)
%PREDICTIVE_CUE_P_VALUE Randomized predictive p-value for a cue label.
%
%   p = predictive_cue_p_value(obj, q, u) returns the randomized probability-
%   integral-transform p-value for cue label q under the current predictive cue
%   pmf, using a uniform random variable u in [0,1]. The value is
%       p = F(q-) + u * f(q),
%   where f is the predictive pmf over cue labels and F(q-) is the cumulative
%   mass of the labels below q; the u * f(q) term spreads the atom f(q) into a
%   continuous [F(q-), F(q)] so that, under a correct model, p is uniform.
%
%   A cue label q that has never been observed is treated as the next novel
%   label (see peekCueLabel), which carries zero mass under the current pmf, so
%   its p-value reduces to F(q-). If q is empty (the default) the p-value is
%   undefined and NaN is returned. If u is not provided it defaults to rand.
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
    fMinus = sum(pmf(labels < qLabel));
    p = min(max(fMinus + u .* f, 0), 1);
end