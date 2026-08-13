function d = jeffreysFiniteClip(d)
%JEFFREYSFINITECLIP Finite-clip epilogue shared by the Jeffreys divergences.
%
%   d = jeffreysFiniteClip(d) applies the common tail of the Jeffreys helpers
%   (gaussianJeffreys, gaussianJeffreysMulti, categoricalJeffreys): a non-finite
%   divergence is replaced by the finite sentinel realmax, then the result is
%   clipped at zero because a divergence is non-negative. This factors out the
%   identical epilogue those three functions share; the behaviour is unchanged.

    if ~isfinite(d)
        d = realmax;  % sentinel: "infinitely divergent" but still finite
    end
    d = max(d, 0);    % divergence is non-negative; clip round-off below 0
end
