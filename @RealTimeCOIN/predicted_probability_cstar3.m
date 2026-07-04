function p = predicted_probability_cstar3(obj)
%PREDICTED_PROBABILITY_CSTAR3 Highest predicted context probability (current trial).
%
%   p = predicted_probability_cstar3(obj) returns the particle-average of the
%   maximum predicted context probability, i.e. the predicted probability of the
%   most likely context (c*3) on the current trial. Mirrors COIN's
%   plot_predicted_probability_cstar3. Returns a scalar in [0, 1] for both the
%   scalar and multi-dimensional models. Reflects the model state as of obj.trial.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    p = mean(max(obj.D.predicted_probabilities, [], 1));
end
