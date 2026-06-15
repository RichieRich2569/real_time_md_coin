function p = predictive_cue_p_value(obj, q, u)
    if nargin < 3
        u = rand;
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
