function p = predicted_probability_cstar1(obj)
%PREDICTED_PROBABILITY_CSTAR1 Predicted probability of the highest-responsibility context.
%
%   p = predicted_probability_cstar1(obj) selects, in each particle, the context
%   with the highest responsibility (c*1) and reads off its predicted
%   probability, then averages over particles. Mirrors COIN's
%   plot_predicted_probability_cstar1. Returns a scalar in [0, 1] for both the
%   scalar and multi-dimensional models (context probabilities are
%   dimension-independent). Reflects the model state as of obj.trial.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    P = obj.num_particles;
    [~, idx] = max(obj.D.responsibilities, [], 1);
    lin = sub2ind(size(obj.D.predicted_probabilities), idx, 1:P);
    p = mean(obj.D.predicted_probabilities(lin));
end
