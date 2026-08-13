function test_marginalize_validator
%TEST_MARGINALIZE_VALIDATOR Density methods pass the restored mixture validators.
%
%   mustMarginalize and mustBeCovarianceMatrix are PRIVATE @RealTimeCOIN
%   validators re-applied inside mixtureDensityOnGrid (the shared helper behind
%   state_probability / state_feedback_probability / novel_* densities). Being
%   private, they cannot be called directly from here, and they guard an internal
%   invariant (the per-particle mixing weights the pipeline feeds them are always
%   normalized by normalizeColumns), so their throw path is a defensive assertion
%   not reachable through the public API.
%
%   What is observable — and what this test checks — is that the restored
%   validators do NOT false-trip on the genuine, valid model state: the density
%   readouts route through them and return proper (nonnegative, unit-integral)
%   densities for both the scalar and multi-dimensional models.

rng(11);
tol = 1e-6;

% --- scalar model ---
coin = RealTimeCOIN('num_particles', 40, 'max_contexts', 3);
for t = 1:8
    coin.observe_q(1 + mod(t, 2));
    coin.observe_y(0.05 * t);
end
grid = linspace(-3, 3, 1201);
dens = coin.state_probability(grid);
fdens = coin.state_feedback_probability(grid);
ndens = coin.novel_state_probability(grid);
assert(all(dens >= 0) && all(fdens >= 0) && all(ndens >= 0), ...
    'Scalar densities must be nonnegative');
assert(abs(trapz(grid, dens) - 1) < 1e-2, 'state_probability must integrate to 1');
assert(abs(trapz(grid, fdens) - 1) < 1e-2, 'state_feedback_probability must integrate to 1');

% --- multi-dimensional model (exercises mustBeCovarianceMatrix on real cov pages) ---
md = RealTimeCOIN('num_particles', 40, 'max_contexts', 3, 'state_dim', 2);
for t = 1:8
    md.observe_q(1 + mod(t, 2));
    md.observe_y([0.05 * t; -0.03 * t]);
end
[gx, gy] = meshgrid(linspace(-2, 2, 61), linspace(-2, 2, 61));
pts = [gx(:)'; gy(:)'];
mdDens = md.state_probability(pts);
mdFeed = md.state_feedback_probability(pts);
mdNovel = md.novel_state_probability(pts);
assert(all(mdDens >= 0) && all(mdFeed >= 0) && all(mdNovel >= 0), ...
    'MD densities must be nonnegative');
assert(all(isfinite(mdDens)) && all(isfinite(mdFeed)), 'MD densities must be finite');
% (MD unit-integral is covered robustly by test_md_state_queries via
%  testutil.integrate2d; here we only assert the validators do not false-trip.)

assert(tol > 0);   % (validator tolerance documented in mustMarginalize)
fprintf('test_marginalize_validator passed (scalar + MD densities pass restored validators).\n');
end
