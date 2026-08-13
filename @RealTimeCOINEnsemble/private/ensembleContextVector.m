function p = ensembleContextVector(obj, memberFn)
%ENSEMBLECONTEXTVECTOR Zero-fill cross-run average of a context probability row.
%   p = ensembleContextVector(obj, memberFn) aligns each member's
%   1-by-(max_contexts+1) context probability vector (returned by memberFn, e.g.
%   @(m) m.responsibilities_vector()) onto the common reference frame and averages
%   with the ZERO-FILL rule of docs/SPEC_ensemble.md Part 10.3: a run that lacks a
%   reference context contributes 0 to that slot, so the result conserves total
%   probability and sums to 1. Real mass goes to reference slots 1..Kref via the
%   per-member matching; each member's novel mass (its slot K_r+1, plus any
%   residual beyond) goes to the reference novel slot Kref+1.
    ali = ensembleContextAlignment(obj);
    R = obj.runs;
    Cmax = obj.members{1}.max_contexts + 1;
    Kref = ali.Kref;
    novelSlot = Kref + 1;

    acc = zeros(1, Cmax);
    for r = 1:R
        v = memberFn(obj.members{r});   % 1-by-Cmax in member r's frame
        Kr = ali.K(r);
        % Real contexts -> matched reference slots.
        for i = 1:Kr
            acc(ali.perm{r}(i)) = acc(ali.perm{r}(i)) + v(i);
        end
        % Member novel slot (K_r+1) and any residual beyond -> reference novel slot.
        if Kr + 1 <= Cmax
            acc(novelSlot) = acc(novelSlot) + sum(v(Kr + 1:end));
        end
    end
    p = acc ./ R;
end
