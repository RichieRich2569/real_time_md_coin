function x = dirichletSample(obj, alpha)
    alpha = alpha(:);
    draws = obj.gammaSample(alpha);
    if sum(draws) <= 0
        draws = zeros(size(alpha));
        first = find(alpha > 0, 1);
        if isempty(first)
            first = 1;
        end
        draws(first) = 1;
    end
    x = draws ./ sum(draws);
end
