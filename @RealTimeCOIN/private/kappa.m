function k = kappa(obj)
%KAPPA Self-transition (stickiness) concentration for the HDP prior.
%   k = kappa(obj) returns the sticky-HDP self-transition mass
%       kappa = alpha_context * rho_context / (1 - rho_context),
%   the extra prior weight placed on staying in the current context (Heald et
%   al. COIN model). The denominator is floored at realmin so that
%   rho_context -> 1 yields a large finite kappa instead of a divide-by-zero.
    k = obj.alpha_context * obj.rho_context / max(1 - obj.rho_context, realmin);
end
