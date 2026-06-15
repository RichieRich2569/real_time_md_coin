function k = kappa(obj)
    k = obj.alpha_context * obj.rho_context / max(1 - obj.rho_context, realmin);
end
