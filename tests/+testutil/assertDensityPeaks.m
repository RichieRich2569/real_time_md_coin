function assertDensityPeaks(name, dmap, grid, means)
%TESTUTIL.ASSERTDENSITYPEAKS Assert each per-context density peaks at its mean.
%   testutil.assertDensityPeaks(name, dmap, grid, means) checks that, for every
%   context key in the containers.Map dmap, the density peaks at the grid point
%   nearest that context's prototype mean (means(context)). The peak must lie
%   within two grid spacings of the expected mean.
    ks = dmap.keys;
    dx = grid(2) - grid(1);
    for i = 1:numel(ks)
        c = ks{i};
        d = dmap(c);
        [~, iPeak] = max(d);
        if abs(grid(iPeak) - means(c)) > 2*dx
            error('testutil:assertDensityPeaks:failed', ...
                'FAILED: %s context %d peak at %.5g, expected mean %.5g', ...
                name, c, grid(iPeak), means(c));
        end
    end
end
