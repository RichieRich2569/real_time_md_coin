function sampleDynamicsMD(obj)
% =========================================================================
% MATHEMATICAL PROOF: CONJUGATE UPDATE FOR MATRIX-NORMAL REGRESSION
% =========================================================================
% Model the per-context dynamics as the matrix linear regression
%     s_i = Theta x_{i-1} + w_i,   w_i ~ N(0, Q),   x_{i-1} = [s_{i-1}; 1],
% with Theta = [A | d] (N x (N+1)). The matrix-normal prior is
%     p(Theta) = MN(M_0, U = Q, V_0),
% whose exponent (cyclic-trace form) is
%     -0.5 Tr[ Q^-1 (Theta - M_0) V_0^-1 (Theta - M_0)' ].
% The Gaussian likelihood exponent across trials (fixed row covariance Q) is
%     -0.5 Tr[ Q^-1 sum_i (s_i - Theta x_{i-1})(s_i - Theta x_{i-1})' ].
% Expanding and collecting the Theta-quadratic and Theta-linear terms with
%     Lambda_xx = sum_i x_{i-1} x_{i-1}',   Lambda_yx = sum_i s_i x_{i-1}',
% then matching to MN(M_post, Q, V_post) gives
%     V_post^-1 = V_0^-1 + Lambda_xx
%     M_post    = (M_0 V_0^-1 + Lambda_yx) V_post.
%
% REDUCTION TO THE SCALAR MODEL (N == 1): with the prior built in
% dynamicsPriorMD.m, V_0^-1 = sigma^2 diag([prec_ret, prec_drift]) and U = Q
% = sigma^2. Then Cov([a;d]) = U V_post = (priorPrec + Lambda_xx/sigma^2)^-1,
% which is exactly covar = inv(priorPrec + ss2/qVar) in sampleDynamics.m, and
% the means coincide likewise. This is what keeps the default behaviour
% unchanged.
% =========================================================================
%
%SAMPLEDYNAMICSMD Sample Theta = [A | d] from the matrix-normal posterior.

    Cmax = obj.max_contexts + 1;                % context slots incl. novel context
    P = obj.num_particles;

    [M0, V0inv, ~] = obj.dynamicsPriorMD();     % prior mean and column precision
    Q = obj.processNoiseCov();                  % process-noise (row) covariance U

    for p = 1:P
        for c = 1:Cmax
            Lxx = obj.D.Lambda_xx(:, :, c, p);  % sum_i x_{i-1} x_{i-1}'
            Lyx = obj.D.Lambda_yx(:, :, c, p);  % sum_i s_i x_{i-1}'

            Vpost = obj.safeInverse(V0inv + Lxx);
            Vpost = (Vpost + Vpost') ./ 2;       % re-symmetrize against round-off
            Mpost = (M0 * V0inv + Lyx) * Vpost;

            % Draw Theta from MN(Mpost, Q, Vpost), rejecting until spectral
            % radius of A < 1 so the sampled dynamics are stable.
            obj.D.Theta(:, :, c, p) = obj.sampleStableTheta(Mpost, Q, Vpost);
        end
    end
end
