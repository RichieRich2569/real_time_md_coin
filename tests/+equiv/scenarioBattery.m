function scenarios = scenarioBattery()
%SCENARIOBATTERY Deterministic (seed, config, inputs) cases for equivalence tests.
%
%   scenarios = equiv.scenarioBattery() returns a struct array covering the
%   scalar and multi-dimensional RealTimeCOIN pipelines with varied context
%   caps, particle counts, cue sequences, and missing-observation patterns.
%   Every scenario carries a FIXED input stream (cues q and feedback y),
%   generated here from a dedicated input seed so that the identical inputs are
%   replayed against both the reference (main) and the candidate (experimental)
%   model code. The model's own RNG is seeded separately in equiv.captureRun via
%   scenario.seed, so both branches consume the same random stream.
%
%   Fields per scenario:
%     name   - char identifier
%     seed   - rng seed for the MODEL's internal sampling (set in captureRun)
%     args   - RealTimeCOIN constructor name-value cell
%     dim    - state dimension (1 for scalar path, >1 for *MD path)
%     q      - 1-by-T cue labels (integer >= 1)
%     y      - dim-by-T feedback; NaN entries denote missing observations
%
%   See also equiv.captureRun, equiv.compareRuns.

    specs = {
        % name              seed  particles  contexts  dim   T    cues  missingFrac
        'scalar_2ctx',        11,   100,        3,      1,    45,   2,    0.00
        'scalar_missing',     12,   100,        3,      1,    45,   2,    0.20
        'scalar_3ctx_small',  13,    40,        4,      1,    50,   3,    0.00
        'md2_basic',          14,    80,        3,      2,    40,   2,    0.00
        'md2_missing',        15,    80,        3,      2,    40,   2,    0.20
        'md3_basic',          16,    60,        4,      3,    35,   2,    0.00
    };

    scenarios = struct('name', {}, 'seed', {}, 'args', {}, 'dim', {}, ...
        'q', {}, 'y', {});
    for i = 1:size(specs, 1)
        [name, seed, nP, nC, dim, T, cues, missFrac] = specs{i, :};
        inputSeed = 7000 + i;                 % distinct from model seed
        [q, y] = makeInputs(inputSeed, dim, T, cues, missFrac);
        scenarios(i).name = name; %#ok<AGROW>
        scenarios(i).seed = seed;
        scenarios(i).args = {'num_particles', nP, 'max_contexts', nC, ...
            'state_dim', dim};
        scenarios(i).dim  = dim;
        scenarios(i).q    = q;
        scenarios(i).y    = y;
    end
end

function [q, y] = makeInputs(inputSeed, dim, T, cues, missFrac)
%MAKEINPUTS Build a deterministic block-structured cue/feedback stream.
    rs = RandStream('twister', 'Seed', inputSeed);

    % Cue sequence: contiguous blocks cycling through the available cues.
    q = zeros(1, T);
    blockLen = max(1, round(T / (2 * cues)));
    label = 1;
    t = 1;
    while t <= T
        stop = min(T, t + blockLen - 1);
        q(t:stop) = label;
        label = mod(label, cues) + 1;
        t = stop + 1;
    end

    % Feedback: per-dimension random walk with occasional block-aligned shifts,
    % so the model must infer more than one context.
    y = zeros(dim, T);
    level = zeros(dim, 1);
    for tt = 1:T
        if tt > 1 && q(tt) ~= q(tt - 1)
            level = level + 0.25 * randn(rs, dim, 1);   % context-change shift
        end
        level = 0.98 * level + 0.02 * randn(rs, dim, 1); % slow drift
        y(:, tt) = level + 0.05 * randn(rs, dim, 1);     % observation noise
    end

    % Inject missing observations (whole-trial NaN) at a fixed cadence.
    if missFrac > 0
        step = max(1, round(1 / missFrac));
        y(:, step:step:T) = NaN;
    end
end
