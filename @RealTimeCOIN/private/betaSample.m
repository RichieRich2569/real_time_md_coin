function b = betaSample(obj, a, bpar)
    x = obj.gammaSample(a);
    y = obj.gammaSample(bpar);
    denom = x + y;
    b = zeros(size(denom));
    good = denom > 0;
    b(good) = x(good) ./ denom(good);
    b(~good) = 1;
end
