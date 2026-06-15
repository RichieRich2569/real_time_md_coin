function d = gaussianJeffreys(~, m1, v1, m2, v2)
    if ~isfinite(v1)
        v1 = 1 ./ eps;
    end
    if ~isfinite(v2)
        v2 = 1 ./ eps;
    end
    v1 = max(v1, eps);
    v2 = max(v2, eps);
    d = 0.5 .* (v1 ./ v2 + v2 ./ v1 + (m1 - m2).^2 .* (1 ./ v1 + 1 ./ v2) - 2);
    if ~isfinite(d)
        d = realmax;
    end
    d = max(d, 0);
end
