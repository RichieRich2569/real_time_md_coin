function [integral, vals] = integrate2d(densFun, center, halfStd)
%TESTUTIL.INTEGRATE2D Tensor-product trapezoidal integral of a 2-D density.
%   [integral, vals] = testutil.integrate2d(densFun, center, halfStd) integrates
%   a 2-D density over a grid spanning center +/- 6*halfStd on each axis.
%   densFun maps an N-by-K matrix of column points to a 1-by-K density row.
%   vals returns the density evaluated on the flattened grid.
n = 121;
span = 6;
ax1 = linspace(center(1) - span*halfStd(1), center(1) + span*halfStd(1), n);
ax2 = linspace(center(2) - span*halfStd(2), center(2) + span*halfStd(2), n);
[G1, G2] = meshgrid(ax1, ax2);
pts = [G1(:)'; G2(:)'];
vals = densFun(pts);
Z = reshape(vals, size(G1));
integral = trapz(ax2, trapz(ax1, Z, 2));
end
