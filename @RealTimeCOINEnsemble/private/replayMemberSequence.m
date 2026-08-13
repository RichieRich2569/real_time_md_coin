function [motor, mu, vout] = replayMemberSequence(generator, R, k, seed, memberParams, qSeq, ySeq, N, T)
%REPLAYMEMBERSEQUENCE Replay one member over a full observation sequence.
%   [motor, mu, vout] = replayMemberSequence(...) constructs a fresh
%   RealTimeCOIN member k under its dedicated substream (makeMemberStream) and
%   drives it through the T-trial sequence (qSeq, ySeq), recording per-trial
%   motor_output and state_moments. It is used by simulate() in both the serial
%   and parfor branches; because the member's stream is a pure function of
%   (seed, k), the recorded traces are identical whichever branch runs, and
%   identical to stepping the live ensemble.
%
%   Returns motor (N-by-T predictive motor output), mu (N-by-T predictive state
%   mean) and vout (predictive state variance: 1-by-T when N == 1, else
%   N-by-N-by-T). The caller's global RNG stream is left unchanged.
    prev = RandStream.getGlobalStream();
    restore = onCleanup(@() RandStream.setGlobalStream(prev));
    RandStream.setGlobalStream(makeMemberStream(generator, R, k, seed));

    m = RealTimeCOIN(memberParams{:});

    motor = nan(N, T);
    mu = nan(N, T);
    if N == 1
        vout = nan(1, T);
    else
        vout = nan(N, N, T);
    end

    for t = 1:T
        m.observe_q(cueAt(qSeq, t));
        m.observe_y(obsAt(ySeq, t));
        motor(:, t) = m.motor_output();
        [mt, vt] = m.state_moments();
        mu(:, t) = mt(:);
        if N == 1
            vout(t) = vt;
        else
            vout(:, :, t) = vt;
        end
    end
end

function q = cueAt(qSeq, t)
%CUEAT Cue for trial t: [] (cue-free) when qSeq is empty, else qSeq(t).
    if isempty(qSeq)
        q = [];
    else
        q = qSeq(t);
    end
end

function y = obsAt(ySeq, t)
%OBSAT Feedback column for trial t: [] when ySeq is empty, else ySeq(:,t).
    if isempty(ySeq)
        y = [];
    else
        y = ySeq(:, t);
    end
end
