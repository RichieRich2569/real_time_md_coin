function results = benchmark_realtimecoin(varargin)
%BENCHMARK_REALTIMECOIN Time RealTimeCOIN over particle/trial grids.

ip = inputParser;
addParameter(ip, 'Trials', 200);
addParameter(ip, 'Particles', [50 100 250]);
addParameter(ip, 'Seed', 3001);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);
particles = cfg.Particles(:)';
elapsed = zeros(size(particles));

cues = 1 + double(rand(1, cfg.Trials) > 0.5);
y = 0.3 * sin((1:cfg.Trials) ./ 20) + 0.05 * randn(1, cfg.Trials);

for i = 1:numel(particles)
    coin = RealTimeCOIN('num_particles', particles(i), 'max_contexts', 5);
    tic;
    for t = 1:cfg.Trials
        coin.observe_q(cues(t));
        coin.observe_y(y(t));
    end
    elapsed(i) = toc;
    fprintf('%d particles x %d trials: %.3f s\n', particles(i), cfg.Trials, elapsed(i));
end

results = struct();
results.particles = particles;
results.trials = cfg.Trials;
results.elapsed_seconds = elapsed;
results.seconds_per_trial = elapsed ./ cfg.Trials;
results.config = cfg;
end
