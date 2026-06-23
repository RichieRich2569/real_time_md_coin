function u = motor_output(obj)
%MOTOR_OUTPUT Expected state feedback marginalised over contexts/particles.
%   Scalar for state_dim == 1; an N-by-1 vector for the multi-dimensional
%   model. The weights are the predicted context probabilities.
    W = obj.D.predicted_probabilities;
    if obj.state_dim == 1
        u = sum(W .* obj.D.state_feedback_mean, 'all') ./ obj.num_particles;
        return;
    end

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
