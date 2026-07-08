function X = normalizeColumns(~, X)
%NORMALIZECOLUMNS Normalise each column of a matrix to sum to 1.
%   X = normalizeColumns(obj, X) rescales every column of X so it sums to 1.
%   A column whose sum is zero or non-finite collapses to the unit vector
%   e_1 (all mass on the first row), which keeps context-weight columns valid
%   even for degenerate particles. The leading obj argument is ignored (private
%   RealTimeCOIN method invoked as obj.normalizeColumns).
    arguments
        ~
        X (:,:) double
    end
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
