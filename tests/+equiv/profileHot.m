function timings = profileHot(varargin)
%PROFILEHOT Time representative hot-path runs of the RealTimeCOIN on path.
%
%   timings = equiv.profileHot() returns median wall-times (via timeit) for a
%   few representative streaming runs, exercising the scalar and multivariate
%   inference pipelines at sizes where the deferred optimizations matter. Run it
%   on the main worktree and on experimental to quantify a Class B change's
%   speedup (governance rule 2: keep a trajectory-changing optimization only if
%   it is materially faster).
%
%   Name-value options:
%     'Trials'    (default 60)   trials per run
%     'Particles' (default 200)  particles per run
%
%   timings is a struct with fields md2, md3, scalar (seconds, median of timeit).
%
%   See also equiv.captureAll, timeit.

    ip = inputParser;
    addParameter(ip, 'Trials', 60);
    addParameter(ip, 'Particles', 200);
    parse(ip, varargin{:});
    T = ip.Results.Trials;
    P = ip.Results.Particles;

    ws = warning('off', 'all');
    cleanup = onCleanup(@() warning(ws));

    timings.scalar = timeit(@() streamRun(1, 3, P, T, 101));
    timings.md2    = timeit(@() streamRun(2, 3, P, T, 102));
    timings.md3    = timeit(@() streamRun(3, 4, P, T, 103));

    fprintf('profileHot (T=%d, P=%d): scalar=%.4fs  md2=%.4fs  md3=%.4fs\n', ...
        T, P, timings.scalar, timings.md2, timings.md3);
end

function streamRun(dim, nC, P, T, seed)
%STREAMRUN One full streaming run over a fixed synthetic input stream.
    rs = RandStream('twister', 'Seed', seed);
    q = 1 + mod(0:T-1, 2);
    y = cumsum(0.1 * randn(rs, dim, T), 2) + 0.05 * randn(rs, dim, T);

    rng(seed, 'twister');
    obj = RealTimeCOIN('num_particles', P, 'max_contexts', nC, 'state_dim', dim);
    for t = 1:T
        obj.observe_q(q(t));
        obj.observe_y(y(:, t));
        obj.motor_output();
    end
end
