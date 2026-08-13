function outMap = ensembleContextDensity(obj, memberFn)
%ENSEMBLECONTEXTDENSITY NaN-omit cross-run average of a per-context density map.
%   outMap = ensembleContextDensity(obj, memberFn) aligns each member's
%   per-context density containers.Map (returned by memberFn, e.g.
%   @(m) m.state_given_context_probability(values)) onto the common reference
%   frame and averages with the NaN-OMIT rule of docs/SPEC_ensemble.md Part 10.3:
%   the density of reference context j is the mean over only the runs that have a
%   context matched to j; if no run has j, key j is absent from the output. Output
%   is a containers.Map keyed by reference label 1..Kref.
    ali = ensembleContextAlignment(obj);
    Kref = ali.Kref;
    outMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    if Kref == 0
        return;
    end

    R = obj.runs;
    sums = cell(1, Kref);
    counts = zeros(1, Kref);
    for r = 1:R
        mp = memberFn(obj.members{r});   % Map keyed by member r's labels 1..K(r)
        Kr = ali.K(r);
        for i = 1:Kr
            if ~isKey(mp, i)
                continue;
            end
            j = ali.perm{r}(i);
            d = mp(i);
            if isempty(sums{j})
                sums{j} = d;
            else
                sums{j} = sums{j} + d;
            end
            counts(j) = counts(j) + 1;
        end
    end

    for j = 1:Kref
        if counts(j) > 0
            outMap(j) = sums{j} ./ counts(j);
        end
    end
end
