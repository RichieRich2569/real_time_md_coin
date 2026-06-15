function sampleParameters(obj)
    obj.sampleGlobalTransitionProbabilities();
    obj.sampleGlobalCueProbabilities();
    obj.sampleDynamics();
    obj.sampleBias();
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();
end
