function u = motor_output(obj)
%MOTOR_OUTPUT Expected state feedback marginalised over contexts/particles.
%   u = motor_output(obj) returns the model's point prediction of the state
%   feedback on the current trial: the mixture mean
%       u = ( sum_{c,p} w_{c,p} * m_{c,p} ) / num_particles
%   where w_{c,p} = D.predicted_probabilities are the predicted context
%   probabilities (per context c, particle p) and m_{c,p} =
%   D.state_feedback_mean are the corresponding per-context state-feedback
%   means.
%
%   For the scalar model (state_dim == 1) u is a scalar; for the
%   multi-dimensional model u is an N-by-1 vector. This is a read-only query:
%   it draws no random numbers and does not mutate particle state.
    arguments
        obj (1, 1) RealTimeCOIN
    end

    W = obj.D.predicted_probabilities;
    if obj.state_dim == 1
        u = sum(W .* obj.D.state_feedback_mean, 'all') ./ obj.num_particles;
        return;
    end

    % Multi-dimensional mixture reduction. Kept as an explicit nested loop
    % (rather than vectorised) so the accumulation order is preserved exactly;
    % see "Deferred optimizations" in the quality review.
    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    u = zeros(N, 1);
    for p = 1:P
        for c = 1:Cmax
            w = W(c, p);
            if w ~= 0
                u = u + w .* obj.D.state_feedback_mean(:, c, p);
            end
        end
    end
    u = u ./ P;
end
