function traces = simulate(obj, qSeq, ySeq)
%SIMULATE Batch-replay a full observation sequence across runs.
%   traces = simulate(obj, qSeq, ySeq) replays a precomputed length-T
%   observation sequence across the R runs and returns per-trial run-averaged
%   traces. This is the offline analogue of COIN.simulate_COIN and the primary
%   parallel-throughput path: when max_cores > 0 the runs are replayed with
%   parfor (capped at max_cores), otherwise serially. Because each run is
%   replayed under its own reproducible substream (a pure function of seed and
%   run index), the result is numerically identical whether serial or parallel,
%   and identical to stepping the ensemble trial-by-trial with observe_q /
%   observe_y and reading the same queries after each trial.
%
%   Inputs:
%     qSeq - 1-by-T cue row (NaN for cue-free trials), or [] for all cue-free.
%     ySeq - feedback: N-by-T (1-by-T for the scalar model); NaN entries mark
%            missing / partially-observed feedback, as for observe_y.
%
%   Output struct traces:
%     .motor_output - N-by-T, column t = motor_output after trial t.
%     .state_mean   - N-by-T, column t = the mu of state_moments after trial t.
%     .state_var    - 1-by-T (scalar model) or N-by-N-by-T, slice t = the v of
%                     state_moments after trial t.
%     .Trial        - 1-by-T equal to 1:T.
%
%   simulate is a one-shot batch on a fresh member set seeded from obj; it does
%   not disturb obj's live stepping state, and repeated calls give identical
%   traces. The caller's global RNG stream is left unchanged.
%
%   See also OBSERVE_Q, OBSERVE_Y, RealTimeCOIN.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        qSeq double = []
        ySeq double = []
    end

    T = sequenceLength(qSeq, ySeq, obj.state_dim_);

    R = obj.runs;
    N = obj.state_dim_;
    generator = obj.rng_generator;
    seed = obj.seed;
    memberParams = obj.member_params;

    runMotor = cell(1, R);
    runMu = cell(1, R);
    runV = cell(1, R);

    if obj.max_cores > 0
        parfor (k = 1:R, obj.max_cores)
            [runMotor{k}, runMu{k}, runV{k}] = replayMemberSequence( ...
                generator, R, k, seed, memberParams, qSeq, ySeq, N, T);
        end
    else
        for k = 1:R
            [runMotor{k}, runMu{k}, runV{k}] = replayMemberSequence( ...
                generator, R, k, seed, memberParams, qSeq, ySeq, N, T);
        end
    end

    traces = struct();
    traces.motor_output = averageAcrossRuns(runMotor);
    [traces.state_mean, traces.state_var] = poolMomentTraces(runMu, runV, N, T);
    traces.Trial = 1:T;
end

function T = sequenceLength(qSeq, ySeq, N)
%SEQUENCELENGTH Resolve and cross-check the sequence length T.
    Ty = size(ySeq, 2);
    if ~isempty(ySeq) && size(ySeq, 1) ~= N
        error('RealTimeCOINEnsemble:simulate:ySize', ...
            'ySeq must have state_dim (%d) rows.', N);
    end
    Tq = numel(qSeq);
    if ~isempty(qSeq) && ~isempty(ySeq) && Tq ~= Ty
        error('RealTimeCOINEnsemble:simulate:lengthMismatch', ...
            'qSeq (%d) and ySeq (%d) imply different sequence lengths.', Tq, Ty);
    end
    if ~isempty(ySeq)
        T = Ty;
    else
        T = Tq;
    end
end

function [meanTrace, varTrace] = poolMomentTraces(runMu, runV, N, T)
%POOLMOMENTTRACES Per-trial pooled-mixture mean/variance across runs.
    R = numel(runMu);
    meanTrace = nan(N, T);
    if N == 1
        varTrace = nan(1, T);
    else
        varTrace = nan(N, N, T);
    end
    mus = cell(1, R);
    vs = cell(1, R);
    for t = 1:T
        for k = 1:R
            mus{k} = runMu{k}(:, t);
            if N == 1
                vs{k} = runV{k}(t);
            else
                vs{k} = runV{k}(:, :, t);
            end
        end
        [mt, vt] = poolMoments(mus, vs, N);
        meanTrace(:, t) = mt;
        if N == 1
            varTrace(t) = vt;
        else
            varTrace(:, :, t) = vt;
        end
    end
end
