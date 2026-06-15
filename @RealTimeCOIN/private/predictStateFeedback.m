function predictStateFeedback(obj)
    obj.D.state_feedback_mean = obj.D.state_mean + obj.D.bias;
    obj.D.state_feedback_var = obj.D.state_var + obj.observationVariance();
end
