function test_static_helpers
%TEST_STATIC_HELPERS Unit tests for public static helper methods.

rng(4);

x = linspace(-6, 6, 2001);
p = RealTimeCOIN.normal_pdf(x, 0, 1);
assert(abs(trapz(x, p) - 1) < 1e-4, 'Normal PDF should integrate to one');

logP = log([0.2 0.3 0.5; 0.1 0.8 0.1]);
lse = RealTimeCOIN.log_sum_exp(logP, 1);
assert(max(abs(exp(lse) - sum(exp(logP), 1))) < 1e-12, 'log_sum_exp mismatch');

T = [0.9 0.1; 0.2 0.8];
piT = RealTimeCOIN.stationary_distribution(T);
assert(max(abs(piT*T - piT)) < 1e-10, 'Stationary distribution is not stationary');
assert(abs(sum(piT) - 1) < 1e-12, 'Stationary distribution is not normalized');

idx = RealTimeCOIN.systematic_resampling([0 0.25 0.75]);
assert(numel(idx) == 3 && all(idx >= 1) && all(idx <= 3), 'Resampling returned invalid indices');

tables = RealTimeCOIN.sample_num_tables(ones(2), [0 3; 2 5]);
assert(all(tables(:) >= 0), 'Table counts must be nonnegative');
assert(all(tables(:) <= [0;2;3;5]), 'Table counts must not exceed observations');
end
