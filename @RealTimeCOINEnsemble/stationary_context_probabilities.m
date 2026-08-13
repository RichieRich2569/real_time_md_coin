function p = stationary_context_probabilities(obj)
%STATIONARY_CONTEXT_PROBABILITIES Cross-run averaged stationary context dist.
%   p = stationary_context_probabilities(obj) returns the 1-by-Kref stationary
%   context distribution averaged across runs in the common reference frame
%   (docs/SPEC_ensemble.md Part 10), or [] when no member holds any context. Each
%   member's 1-by-K_r stationary distribution is zero-filled onto the reference
%   contexts via the cross-run matching, summed across runs and renormalised to
%   sum to 1. Read-only.
%
%   See also RESPONSIBILITIES_VECTOR, RealTimeCOIN/stationary_context_probabilities.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    ali = ensembleContextAlignment(obj);
    Kref = ali.Kref;
    if Kref == 0
        p = [];
        return;
    end

    acc = zeros(1, Kref);
    for r = 1:obj.runs
        s = obj.members{r}.stationary_context_probabilities();   % 1-by-K(r) or []
        if isempty(s)
            continue;
        end
        Kr = min(ali.K(r), numel(s));
        for i = 1:Kr
            acc(ali.perm{r}(i)) = acc(ali.perm{r}(i)) + s(i);
        end
    end

    total = sum(acc);
    if total > 0
        p = acc ./ total;
    else
        p = ones(1, Kref) ./ Kref;
    end
end
