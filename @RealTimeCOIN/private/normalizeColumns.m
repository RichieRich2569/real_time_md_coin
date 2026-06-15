function X = normalizeColumns(~, X)
    sums = sum(X, 1);
    for p = 1:size(X,2)
        if sums(p) > 0 && isfinite(sums(p))
            X(:,p) = X(:,p) ./ sums(p);
        else
            X(:,p) = 0;
            X(1,p) = 1;
        end
    end
end
