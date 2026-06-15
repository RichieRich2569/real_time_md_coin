function z = safeDivide(~, a, b)
    z = zeros(size(a));
    good = abs(b) > eps;
    z(good) = a(good) ./ b(good);
end
