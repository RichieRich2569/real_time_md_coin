function g = gammaSample(~, shape)
    g = zeros(size(shape));
    good = shape > 0;
    if any(good, 'all')
        g(good) = randg(shape(good));
    end
end
