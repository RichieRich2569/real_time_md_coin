%[text] # RealTimeCOIN - multi-dimensional examples
%[text] Companion notebook: `scalar_examples.m` (the scalar cases).
%[text] ## Section 1 - Setup
%[text] Put the RealTimeCOIN class (repo root) and the coinviz package (this folder) on the path. Run this section once before the others.
% Locate the repo root by looking for its folders (@RealTimeCOIN and
% examples/+coinviz). We can't trust mfilename here: pressing "Run Section"
% in the MATLAB Editor executes a copy of the code from a temp folder, so
% mfilename('fullpath') points at that temp path, not this file.
here = fileparts(mfilename('fullpath'));
repoRoot = '';
for c = {here, fileparts(here), pwd, fileparts(pwd)}
    d = c{1};
    if ~isempty(d) && isfolder(fullfile(d, '@RealTimeCOIN')) ...
            && isfolder(fullfile(d, 'examples', '+coinviz'))
        repoRoot = d; break;
    end
end
if isempty(repoRoot)
    error(['Cannot locate the RealTimeCOIN repo. cd to the repo root ' ...
        '(the folder containing @RealTimeCOIN and examples/) and re-run this section.']);
end
addpath(repoRoot);                                % RealTimeCOIN class
addpath(fullfile(repoRoot, 'examples'));          % +coinviz package
fprintf('Paths added. RealTimeCOIN on path: %d\n', exist('RealTimeCOIN', 'class') == 8); %[output:0166c791]
%%
%[text] ## Section 2 - Small 2-D demo with correlated noise
%[text] A 2-D model with full (correlated) process and observation noise covariances. Two cues point at two latent targets. Before each observation we read the one-step predictive feedback distribution (predictive\_feedback\_moments) and draw it as a +/-1 sigma band; after each we read the posterior state moments.
rng(7);
N = 2;
Q = [2.5e-3, 1.0e-3; 1.0e-3, 2.0e-3];
R = [9.0e-3, -2.0e-3; -2.0e-3, 1.2e-2];
coin = RealTimeCOIN('num_particles', 300, 'max_contexts', 4, 'state_dim', N, ...
    'infer_bias', true, 'process_noise_covariance', Q, 'observation_noise_covariance', R);

cues = [1 1 1 2 2 2 2 1 1 2];
targets = containers.Map('KeyType', 'double', 'ValueType', 'any');
targets(1) = [ 0.5; -0.2];
targets(2) = [-0.3;  0.6];

T = numel(cues);
trueState = zeros(N, T); feedbacks = zeros(N, T);
predMean = zeros(N, T); predStd = zeros(N, T); postMean = zeros(N, T);
Lr = chol(R, 'lower');
for t = 1:T
    trueState(:, t) = targets(cues(t));
    feedbacks(:, t) = trueState(:, t) + Lr * randn(N, 1);

    coin.observe_q(cues(t));
    [mu, Sigma] = coin.predictive_feedback_moments(cues(t));   % read-only prediction
    predMean(:, t) = mu; predStd(:, t) = sqrt(max(diag(Sigma), 0));

    coin.observe_y(feedbacks(:, t));
    postMean(:, t) = coin.state_moments();                      % N-by-1 posterior mean
end

coinviz.stateTrace(1:T, feedbacks, trueState, predMean, [], 'FigName', 'MD: small 2-D demo', ...
    'Band', predStd, 'PredictedName', 'Predictive \pm1\sigma');
coinviz.trajectory2D(postMean, [targets(1), targets(2)], 'FigName', 'MD: 2-D trajectory', ...
    'Title', 'Posterior state trajectory', 'PathName', 'Posterior mean', ...
    'TargetName', 'Targets', 'ObservedXY', feedbacks);
%%
%[text] ## Section 3 - Long 2-D cued run with recall, densities and the novel context
%[text] 500 trials over four cued 2-D "contingencies" (distinct latent targets), including a **recall** of context 1 to show a previously learned contingency is re-engaged rather than relearned. At the end we draw per-context density heat-maps in both feedback and state space, including the novel (not-yet-instantiated) context.
rng(11);
N = 2; maxContexts = 6;
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', maxContexts, ...
    'state_dim', N, 'infer_bias', true);

targets   = [ 0.6 -0.4  0.2  0.6 ;
             -0.3  0.5  0.6 -0.3 ];
blockCues = [1 2 3 1];
blockLens = [150 150 130 70];
T = sum(blockLens);
cues = zeros(1, T); trueState = zeros(N, T);
bnd = [0 cumsum(blockLens)];
for b = 1:numel(blockLens)
    idx = (bnd(b)+1):bnd(b+1);
    cues(idx) = blockCues(b);
    trueState(:, idx) = repmat(targets(:, b), 1, numel(idx));
end
feedbacks = trueState + coin.sigma_sensory_noise * 0.01 * randn(N, T);

Cwidth = maxContexts + 1;
motorOutput = zeros(N, T); postStateMean = zeros(N, T);
prevCtx = zeros(T, Cwidth); postCtx = zeros(T, Cwidth);
fprintf('Running 500-trial 2-D cued example...\n'); tic;
for t = 1:T
    coin.observe_q(cues(t));
    motorOutput(:, t) = coin.predictive_motor_output(cues(t));     % expected next feedback
    pv = coin.predicted_context_probabilities_local();
    prevCtx(t, 1:numel(pv)) = pv;

    coin.observe_y(feedbacks(:, t));
    postStateMean(:, t) = coin.state_moments();
    rv = coin.context_responsibilities_local();
    postCtx(t, 1:numel(rv)) = rv;
end
fprintf('Done in %.2f s. Sampled-context occupancy: [%s]\n', toc, ...
    num2str(coin.sampled_context_count(), '%.2f '));

blockEdges = bnd(2:end-1);
coinviz.stateTrace(1:T, feedbacks, trueState, motorOutput, blockEdges, ...
    'FigName', 'MD: motor output vs truth', 'PredictedName', 'Motor output');
coinviz.contextBars(prevCtx, postCtx, 'FigName', 'MD: context probabilities', ...
    'NovelContext', true);
coinviz.trajectory2D(motorOutput, targets, 'FigName', 'MD: motor-output path', ...
    'Title', 'Motor-output path and block targets', 'PathName', 'Motor output');

% Per-context densities incl. the novel context, in feedback and state space.
D = coin.diagnostics();
[~, dominantCue] = max(D.cue_prob, [], 2);
ctxKeys = cell2mat(coin.state_given_context_probability(targets(:, 1)).keys);
ctxTargets = zeros(N, numel(ctxKeys));
for i = 1:numel(ctxKeys)
    bcol = find(blockCues == dominantCue(ctxKeys(i)), 1);
    if isempty(bcol), bcol = 1; end
    ctxTargets(:, i) = targets(:, bcol);
end
coinviz.densityHeatmaps(coin, D, ctxKeys, ctxTargets, 'Space', 'feedback', 'CueLabels', dominantCue);
coinviz.densityHeatmaps(coin, D, ctxKeys, ctxTargets, 'Space', 'state',    'CueLabels', dominantCue);

% Inferred per-context parameters (multi-dimensional): retention matrix A,
% drift and bias vectors, plus the transition and cue prototypes.
fprintf('\nInferred MD contexts (K = %d):\n', D.K);
for i = 1:D.K
    fprintf('  ctx %d: drift [% .3f % .3f]  bias [% .3f % .3f]  state_mean [% .3f % .3f]\n', ...
        i, D.drift(1, i), D.drift(2, i), D.bias(1, i), D.bias(2, i), ...
        D.state_mean(1, i), D.state_mean(2, i));
    fprintf('         A = [% .3f % .3f ; % .3f % .3f]\n', ...
        D.A(1, 1, i), D.A(1, 2, i), D.A(2, 1, i), D.A(2, 2, i));
end
fprintf('  transition_prob (K x K+1):\n'); disp(D.transition_prob);
fprintf('  cue_prob (K x Q):\n'); disp(D.cue_prob);
%%
%[text] ## Section 4 - N = 4: fixed, drifting and dimension-swapping contingencies
%[text] A higher-dimensional, **cue-free** run over three visually distinct dynamic regimes, then a return to the first: (1) a fixed target, (2) a linearly drifting target, (3) a "swap" regime whose latent dynamics exchange dims 1\<-\>2 each trial. We track an "effective number of contexts" as the count of context slots that have ever carried appreciable responsibility.
rng(21);
N = 4;
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 6, 'state_dim', N, ...
    'infer_bias', false, 'process_noise_covariance', diag([0.05 0.10 0.15 0.20].^2));

blockLens = [90 80 80 90]; T = sum(blockLens);
bnd = [0 cumsum(blockLens)];
trueState = zeros(N, T);
fixedTgt = [0.5; -0.5; 0.5; -0.5];
swapA = [0 1 0 0; 1 0 0 0; 0 0 1 0; 0 0 0 1];       % swaps dims 1 and 2
for b = 1:numel(blockLens)
    idx = (bnd(b)+1):bnd(b+1);
    switch b
        case {1, 4}                                  % fixed target (block 4 = recall)
            trueState(:, idx) = repmat(fixedTgt, 1, numel(idx));
        case 2                                        % linear drift
            k = 0:(numel(idx)-1);
            trueState(:, idx) = fixedTgt + [0.004; -0.003; 0.002; -0.001] * k;
        case 3                                        % dimension-swapping dynamics
            s = fixedTgt;
            for j = 1:numel(idx)
                s = swapA * s;
                trueState(:, idx(j)) = s;
            end
    end
end
feedbacks = trueState + 0.02 * randn(N, T);

Cwidth = coin.max_contexts + 1;
postStateMean = zeros(N, T); predFeedback = zeros(N, T);
postCtx = zeros(T, Cwidth); effContexts = zeros(1, T);
fprintf('Running 340-trial N=4 example...\n');
for t = 1:T
    predFeedback(:, t) = coin.predictive_motor_output();          % no cue -> pending state
    coin.observe_y(feedbacks(:, t));
    postStateMean(:, t) = coin.state_moments();
    rv = coin.context_responsibilities_local();
    postCtx(t, 1:numel(rv)) = rv;
    effContexts(t) = nnz(max(postCtx(1:t, :), [], 1) > 0.05);   % slots ever active so far
end
blockEdges = bnd(2:end-1);

coinviz.stateTrace(1:T, feedbacks, trueState, postStateMean, blockEdges, ...
    'FigName', 'MD N=4: state tracking', 'PredictedName', 'Posterior mean');
coinviz.contextBars([], postCtx, 'FigName', 'MD N=4: context responsibilities', ...
    'PostTitle', 'Context responsibilities (fixed / drift / swap / recall)');
figure('Name', 'MD N=4: effective number of contexts');
plot(1:T, effContexts, 'LineWidth', 1.4, 'Color', coinviz.palette(1));
hold on; yl = ylim();
for e = blockEdges, plot([e e], yl, 'Color', [0.5 0.5 0.5 0.4]); end
xlabel('Trial'); ylabel('Effective # contexts'); title('Contexts discovered over the run');
%%
%[text] ## Section 5 - Rotation / oscillation pattern
%[text] A 2-D latent target that traces a slowly shrinking spiral (a rotation applied each trial). This is a "pattern worth identifying": the observations circle the origin, and the motor output should follow the spiral with a lag. A distinctive trajectory plot results.
rng(31);
N = 2;
coin = RealTimeCOIN('num_particles', 300, 'max_contexts', 4, 'state_dim', N, ...
    'infer_bias', true);

T = 240; omega = 2*pi/60; decay = 0.995;
Rot = [cos(omega) -sin(omega); sin(omega) cos(omega)];
trueState = zeros(N, T); s = [0.8; 0.0];
for t = 1:T
    trueState(:, t) = s;
    s = decay * (Rot * s);
end
feedbacks = trueState + 0.02 * randn(N, T);

motorOutput = zeros(N, T); postMean = zeros(N, T);
Cwidth = coin.max_contexts + 1;   % set from THIS section's model (was carried over)
postCtx = zeros(T, Cwidth);
for t = 1:T
    motorOutput(:, t) = coin.predictive_motor_output();
    coin.observe_y(feedbacks(:, t));
    postMean(:, t) = coin.state_moments();
    rv = coin.context_responsibilities_local();
    postCtx(t, 1:numel(rv)) = rv;
end

% Show Predicted Parameters
info = coin.diagnostics();
fprintf('Retention matrix:\n'); disp(info.A);
fprintf('Drift vector:\n'); disp(info.drift);
fprintf('Bias vector:\n'); disp(info.bias);

coinviz.stateTrace(1:T, feedbacks, trueState, postMean, [], ...
    'FigName', 'MD: oscillation (per dim)', 'PredictedName', 'Posterior mean');
coinviz.trajectory2D(postMean, [], 'FigName', 'MD: spiral trajectory', ...
    'Title', 'Rotation / oscillation pattern', 'PathName', 'Posterior mean', ...
    'ObservedXY', feedbacks);
coinviz.contextBars([], postCtx, 'FigName', 'Rotation/Oscillation pattern context responsibilities', ...
    'PostTitle', 'Context responsibilities');
%%
%[text] ## Section 6 - N = 6 with structured (correlated) covariance
%[text] A larger 6-D model with fully correlated, non-isotropic process and observation covariances. Two cued targets alternate. Besides tracking, we check predictive **calibration** with predictive\_state\_feedback\_cdf: the probability-integral-transform values should be roughly uniform on \[0,1\] if the predictive distribution is well calibrated.
rng(41);
N = 6;
B = 0.04 * (eye(N) + 0.5 * diag(ones(N-1,1), 1) + 0.5 * diag(ones(N-1,1), -1));
Q = B * B';                                          % SPD process covariance
C = 0.06 * (eye(N) + 0.3 * triu(ones(N), 1) + 0.3 * tril(ones(N), -1)) / N;
R = C * C' + 1e-3 * eye(N);                          % SPD observation covariance
coin = RealTimeCOIN('num_particles', 150, 'max_contexts', 4, 'state_dim', N, ...
    'infer_bias', true, 'process_noise_covariance', Q, 'observation_noise_covariance', R);

tgtA = linspace(0.5, -0.5, N)';
tgtB = -tgtA;
blockLens = [70 70 60]; blockCues = [1 2 1]; T = sum(blockLens);
bnd = [0 cumsum(blockLens)];
cues = zeros(1, T); trueState = zeros(N, T);
for b = 1:numel(blockLens)
    idx = (bnd(b)+1):bnd(b+1); cues(idx) = blockCues(b);
    tgt = tgtA; if blockCues(b) == 2, tgt = tgtB; end
    trueState(:, idx) = repmat(tgt, 1, numel(idx));
end
Lr = chol(R, 'lower');
feedbacks = trueState + Lr * randn(N, T);

postMean = zeros(N, T); pit = zeros(N, T);
for t = 1:T
    coin.observe_q(cues(t));
    pit(:, t) = coin.predictive_state_feedback_cdf(feedbacks(:, t), cues(t));  % PIT values
    coin.observe_y(feedbacks(:, t));
    postMean(:, t) = coin.state_moments();
end
blockEdges = bnd(2:end-1);

coinviz.stateTrace(1:T, feedbacks, trueState, postMean, blockEdges, ...
    'FigName', 'MD N=6: state tracking', 'PredictedName', 'Posterior mean');
figure('Name', 'MD N=6: predictive calibration (PIT)');
histogram(pit(:), 10, 'Normalization', 'pdf', 'FaceColor', coinviz.palette(1));
hold on; plot([0 1], [1 1], 'k--', 'LineWidth', 1.2);
xlabel('Probability-integral-transform value'); ylabel('Density');
title('PIT of one-step predictions (flat = well calibrated)');
%%
%[text] ## Section 7 - Correlated dimensions with a strong recall block
%[text] Two 2-D contexts whose dimensions are strongly correlated, presented as A, B, then A again. The point is the recall: when context A returns, the responsibilities should jump back to the original A context rather than spawning a new one. We overlay the responsibility of the first context to make the re-engagement obvious.
rng(51);
N = 2;
Q = [3e-3 2.6e-3; 2.6e-3 3e-3];                      % strongly correlated dims
coin = RealTimeCOIN('num_particles', 300, 'max_contexts', 5, 'state_dim', N, ...
    'infer_bias', true, 'process_noise_covariance', Q);

tgtA = [0.6; 0.5]; tgtB = [-0.5; -0.6];
blockLens = [120 120 120]; whichT = {tgtA, tgtB, tgtA}; T = sum(blockLens);
bnd = [0 cumsum(blockLens)];
cues = zeros(1, T); trueState = zeros(N, T);
for b = 1:numel(blockLens)
    idx = (bnd(b)+1):bnd(b+1);
    cues(idx) = (whichT{b}(1) > 0) + 1;               % cue 1 for A, 2 for B
    trueState(:, idx) = repmat(whichT{b}, 1, numel(idx));
end
feedbacks = trueState + 0.02 * randn(N, T);

Cwidth = coin.max_contexts + 1;
postMean = zeros(N, T); postCtx = zeros(T, Cwidth);
for t = 1:T
    coin.observe_q(cues(t));
    coin.observe_y(feedbacks(:, t));
    postMean(:, t) = coin.state_moments();
    rv = coin.context_responsibilities_local();
    postCtx(t, 1:numel(rv)) = rv;
end
blockEdges = bnd(2:end-1);

coinviz.stateTrace(1:T, feedbacks, trueState, postMean, blockEdges, ...
    'FigName', 'MD: correlated dims + recall', 'PredictedName', 'Posterior mean');
coinviz.contextBars([], postCtx, 'FigName', 'MD: recall responsibilities', ...
    'PostTitle', 'Responsibilities (A, B, A) - watch context 1 re-engage');
coinviz.trajectory2D(postMean, [tgtA, tgtB], 'FigName', 'MD: recall trajectory', ...
    'Title', 'A -> B -> A recall', 'PathName', 'Posterior mean', 'ObservedXY', feedbacks);
%%
%[text] ## Section 8 - MD method coverage (remaining API) and stationary save/load
%[text] Trains a compact 2-D model, then calls every remaining public method in the multi-dimensional setting and prints a checklist, finishing with a set\_stationary / save / load round-trip in MD.
rng(61);
N = 2;
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 4, 'state_dim', N, 'infer_bias', true);
cues = [1 1 2 2 1 2 1 2 1 2]; tgt = containers.Map({1,2}, {[0.5;-0.3], [-0.4;0.5]});
for t = 1:numel(cues)
    coin.observe_q(cues(t));
    coin.observe_y(tgt(cues(t)) + 0.02 * randn(N, 1));
end

fprintf('\n================ MD method coverage ================\n');
fprintf('motor_output ..................... [%s]\n', num2str(coin.motor_output()', '%.3f '));
fprintf('predictive_motor_output(1) ....... [%s]\n', num2str(coin.predictive_motor_output(1)', '%.3f '));
fprintf('predictive_cue_p_value(2, 0.5) ... %.3f\n', coin.predictive_cue_p_value(2, 0.5));
[mS, cS] = coin.state_moments();
fprintf('state_moments .................... mean [%s], cov diag [%s]\n', ...
    num2str(mS', '%.3f '), num2str(diag(cS)', '%.4f '));
fprintf('predicted_context_probabilities .. [%s]\n', num2str(coin.predicted_context_probabilities_vector(), '%.2f '));
fprintf('responsibilities ................. [%s]\n', num2str(coin.responsibilities_vector(), '%.2f '));
fprintf('sampled_context_count ............ [%s]\n', num2str(coin.sampled_context_count(), '%.2f '));
fprintf('sampled_context_count_local ...... [%s]\n', num2str(coin.sampled_context_count_local(), '%.2f '));
fprintf('context_predicted_probabilities .. %d contexts\n', coin.predicted_context_probabilities_map().Count);
fprintf('context_responsibilities ......... %d contexts\n', coin.responsibilities_map().Count);
fprintf('context_alignment ................ K = %d\n', coin.context_alignment().K);

% New per-trial c*/component read-outs (MD returns N-by-1 states).
fprintf('explicit_component ............... [%s]\n', num2str(coin.explicit_component()', '%.3f '));
fprintf('implicit_component ............... [%s]\n', num2str(coin.implicit_component()', '%.3f '));
fprintf('state_cstar1 ..................... [%s]\n', num2str(coin.state_cstar1()', '%.3f '));
fprintf('state_cstar2 ..................... [%s]\n', num2str(coin.state_cstar2()', '%.3f '));
fprintf('state_cstar3 ..................... [%s]\n', num2str(coin.state_cstar3()', '%.3f '));
fprintf('predicted_probability_cstar1/3 ... %.3f / %.3f\n', ...
    coin.predicted_probability_cstar1(), coin.predicted_probability_cstar3());
% Transition / cue / stationary distributions (dimension-independent).
ltp = coin.local_transition_probabilities();
lcp = coin.local_cue_probabilities();
scp = coin.stationary_context_probabilities();
fprintf('local_transition_probabilities ... %d-by-%d (rows sum to 1: %d)\n', ...
    size(ltp, 1), size(ltp, 2), all(abs(sum(ltp, 2) - 1) < 1e-9));
fprintf('local_cue_probabilities .......... %d-by-%d\n', size(lcp, 1), size(lcp, 2));
fprintf('stationary_context_probabilities . [%s]\n', num2str(scp, '%.3f '));
fprintf('global_transition_probabilities .. [%s]\n', num2str(coin.global_transition_probabilities(), '%.2f '));
fprintf('global_cue_probabilities ......... [%s]\n', num2str(coin.global_cue_probabilities(), '%.2f '));

% Scalar-dynamics-only methods must reject an N-dimensional model.
guards = {@() coin.kalman_gain_cstar1(), @() coin.kalman_gain_cstar2(), ...
    @() coin.retention_given_context_probability(0.9), ...
    @() coin.drift_given_context_probability(0.0), ...
    @() coin.bias_given_context_probability(0.0)};
for gi = 1:numel(guards)
    try
        guards{gi}();
        fprintf('  (unexpected: scalar-only method %d did not error)\n', gi);
    catch e
        fprintf('  guarded scalar-only method -> %s\n', e.identifier);
    end
end

% Stationary save / load round-trip (MD).
tmpFile = [tempname '.mat'];
before = coin.responsibilities_vector();
coin.saveModel(tmpFile, true);                       % stationarize + save
reloaded = RealTimeCOIN('state_dim', N, 'infer_bias', true);
reloaded.loadModel(tmpFile);
fprintf('saveModel/loadModel (MD) ......... reloaded Trial = %d (reset by set_stationary)\n', reloaded.Trial);

% set_stationary drives the live model onto the analytic stationary distribution
% of its learned transition matrix (same demonstration as the scalar notebook).
piAnalytic = coin.stationary_context_probabilities();
coin.set_stationary();
fprintf('set_stationary (MD) .............. Trial = %d\n', coin.Trial);
predicted = coin.predicted_context_probabilities_vector();
Kk = numel(piAnalytic);
predKnown = predicted(1:Kk) / sum(predicted(1:Kk));
fprintf('stationary match (MD): analytic [%s] vs adopted [%s], max diff %.3f\n', ...
    num2str(piAnalytic, '%.3f '), num2str(predKnown, '%.3f '), max(abs(piAnalytic - predKnown)));
assert(max(abs(coin.predicted_context_probabilities_vector() - coin.responsibilities_vector())) < 1e-9, ...
    'MD predicted == responsibilities after set_stationary');
delete(tmpFile);
fprintf('===================================================\n');
%%
%[text] ## Section 9 - Missing (NaN) observations in the MD model
%[text] Two kinds of missingness: (a) whole trials where *every* coordinate is NaN (a channel trial - no feedback at all), and (b) trials where only *some* coordinates are observed. During a NaN trial the filter propagates the prior, so the prediction drifts back toward the process mean and the state posterior widens; an observed coordinate still corrects its own dimension. We watch the feedback prediction (with a +/-1 sigma band) and the per-dimension state-probability evolution across the gaps.
rng(90);
N = 2;
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 3, 'state_dim', N, ...
    'infer_bias', true);
tgt = [0.4; -0.3];
T = 45;
allNaN  = 16:22;                     % (a) every coordinate NaN (channel trials)
partial = 30:38;                     % (b) only dim 1 observed
feedbacks = nan(N, T);
predMean = zeros(N, T);
predSd   = zeros(N, T);
sGrid = linspace(-1.0, 1.0, 161);
dens1 = zeros(numel(sGrid), T);
dens2 = zeros(numel(sGrid), T);
prevCtx = zeros(T, coin.max_contexts + 1);
for t = 1:T
    coin.observe_q(1);
    [mu, Sigma] = coin.predictive_feedback_moments(1);
    predMean(:, t) = mu;
    predSd(:, t) = sqrt(diag(Sigma));
    prevCtx(t, :) = coin.predicted_context_probabilities_vector();

    y = tgt + 0.03 * randn(N, 1);
    if ismember(t, allNaN)
        y = [NaN; NaN];              % all coordinates missing
    elseif ismember(t, partial)
        y(2) = NaN;                  % dim 2 unobserved
    end
    feedbacks(:, t) = y;
    coin.observe_y(y);

    % Per-dimension marginal state density (slice the other dim at its mean).
    [mm, ~] = coin.state_moments();
    dens1(:, t) = coin.state_probability([sGrid; mm(2) * ones(1, numel(sGrid))])';
    dens2(:, t) = coin.state_probability([mm(1) * ones(1, numel(sGrid)); sGrid])';
end
trueState = repmat(tgt, 1, T);
missTrials = [allNaN, partial];
coinviz.stateTrace(1:T, feedbacks, trueState, predMean, [], ...
    'FigName', 'MD NaN: prediction through missing trials', ...
    'Band', predSd, 'PredictedName', 'Predicted feedback');
coinviz.densityEvolution(1:T, sGrid, dens1, ...
    'FigName', 'MD NaN: dim 1 state evolution', ...
    'Title', 'p(state dim 1) over trials (missing trials dashed)', ...
    'YLabel', 'State dim 1', 'MissingTrials', missTrials);
coinviz.densityEvolution(1:T, sGrid, dens2, ...
    'FigName', 'MD NaN: dim 2 state evolution', ...
    'Title', 'p(state dim 2) over trials (missing trials dashed)', ...
    'YLabel', 'State dim 2', 'MissingTrials', missTrials);
coinviz.contextBars(prevCtx, [], 'FigName', 'MD NaN: context probabilities', ...
    'NovelContext', true, 'PrevTitle', 'Predicted context probs across missing trials');
fprintf('MD NaN section: all-NaN trials %d-%d, partial (dim 1 only) %d-%d\n', ...
    allNaN(1), allNaN(end), partial(1), partial(end));
assert(all(isfinite(predMean(:))) && all(isfinite(dens1(:))) && all(isfinite(dens2(:))), ...
    'predictions and densities remain finite through NaN trials');
%%
%[text] ## Section 10 - MD prior / covariance exploration
%[text] The same 2-D cued sequence shown to two models with different observation-noise priors: isotropic versus strongly anti-correlated. The correlated prior couples the two dimensions during the Kalman update, so the same feedback pulls the estimate along a different direction. We compare the motor-output trajectories and the number of inferred contexts.
rng(100);
N = 2;
cues = [1 1 2 2 1 2 1 2 1 2 1 2];
tgtMap = containers.Map({1, 2}, {[0.5; 0.4], [-0.4; -0.5]});
Tt = numel(cues);
fb = zeros(N, Tt);
for t = 1:Tt, fb(:, t) = tgtMap(cues(t)) + 0.03 * randn(N, 1); end

Riso  = 1e-3 * eye(N);
Rcorr = [1e-3, -0.8e-3; -0.8e-3, 1e-3];      % SPD, strongly anti-correlated
isoModel  = RealTimeCOIN('state_dim', N, 'max_contexts', 3, 'observation_noise_covariance', Riso);
corrModel = RealTimeCOIN('state_dim', N, 'max_contexts', 3, 'observation_noise_covariance', Rcorr);
isoPath = zeros(N, Tt); corrPath = zeros(N, Tt);
for t = 1:Tt
    isoModel.observe_q(cues(t));  isoModel.observe_y(fb(:, t));
    corrModel.observe_q(cues(t)); corrModel.observe_y(fb(:, t));
    isoPath(:, t)  = isoModel.motor_output();
    corrPath(:, t) = corrModel.motor_output();
end
targets = [tgtMap(1), tgtMap(2)];
coinviz.trajectory2D(isoPath, targets, 'FigName', 'MD prior: isotropic R', ...
    'Title', 'Isotropic observation noise', 'ObservedXY', fb);
coinviz.trajectory2D(corrPath, targets, 'FigName', 'MD prior: correlated R', ...
    'Title', 'Anti-correlated observation noise', 'ObservedXY', fb);
fprintf('Isotropic contexts: %d;  Correlated contexts: %d\n', ...
    isoModel.diagnostics().K, corrModel.diagnostics().K);
%%
%[text] ## Section 11 - Parallel runs: the ensemble (multi-dimensional)
%[text] `RealTimeCOINEnsemble` wraps R independent N-dimensional `RealTimeCOIN` filters fed the same vector-valued feedback and returns their equal-weight average. `simulate` batch-replays every run (with `parfor` when `max_cores > 0`) and returns the per-trial run-averaged `motor_output` (N-by-T) plus the pooled `state_mean` / `state_var`; context-aligned averages (responsibilities, per-context densities) work exactly as in the scalar notebook. Here a 2-D A/B/A recall schedule is run over 30 members.
rng(21);
N = 2;
tgtA = [0.4; -0.3];  tgtB = [-0.35; 0.25];
blockLen = 30;
targetsSeq = [repmat(tgtA, 1, blockLen), repmat(tgtB, 1, blockLen), repmat(tgtA, 1, blockLen)];
cues = [ones(1, blockLen), 2 * ones(1, blockLen), ones(1, blockLen)];
T = size(targetsSeq, 2);
obs = targetsSeq + 0.03 * randn(N, T);

% Batch replay across runs (parfor when max_cores > 0), returning averaged traces.
ens = RealTimeCOINEnsemble('runs', 30, 'seed', 3, 'max_cores', 8, ...
    'state_dim', N, 'max_contexts', 4, 'num_particles', 100);
tr = ens.simulate(cues, obs);
fprintf('Ensemble (MD): runs = %d, motor trace = %s\n', ens.runs, mat2str(size(tr.motor_output))); %[output:6eb6090c]
coinviz.trajectory2D(tr.motor_output, [tgtA, tgtB], ... %[output:group:1e6d2439] %[output:485352a6]
    'FigName', 'Ensemble MD: run-averaged trajectory', ... %[output:485352a6]
    'Title', 'Run-averaged motor\_output (30 runs, parfor)', 'ObservedXY', obs); %[output:group:1e6d2439] %[output:485352a6]

% Live stepping to read context-aligned ensemble summaries at the final trial.
ensLive = RealTimeCOINEnsemble('runs', 30, 'seed', 3, 'state_dim', N, ...
    'max_contexts', 4, 'num_particles', 100);
for t = 1:T, ensLive.observe_q(cues(t)); ensLive.observe_y(obs(:, t)); end
resp = ensLive.responsibilities_vector();
[muE, covE] = ensLive.state_moments();
fprintf('Run-averaged responsibilities: [%s] (sum %.3f)\n', num2str(resp, '%.2f '), sum(resp)); %[output:7c560aef]
fprintf('Pooled state mean = [% .3f % .3f]\n', muE(1), muE(2)); %[output:415ebfed]
fprintf('Pooled state covariance:\n'); disp(covE); %[output:6df3ce89]

% Per-context 2-D state density (reference frame), averaged across runs.
gx = linspace(-0.8, 0.8, 61);
[GX, GY] = meshgrid(gx, gx);
gridPts = [GX(:)'; GY(:)'];
dmap = ensLive.state_given_context_probability(gridPts);   % Map: ctx -> 1 x numel(GX)
ctxKeys = cell2mat(dmap.keys);
figure('Name', 'Ensemble MD: per-context state density'); %[output:0067be03]
tiledlayout(1, max(numel(ctxKeys), 1), 'TileSpacing', 'compact'); %[output:0067be03]
for i = 1:numel(ctxKeys)
    nexttile; %[output:0067be03]
    imagesc(gx, gx, reshape(dmap(ctxKeys(i)), size(GX)));
    set(gca, 'YDir', 'normal'); axis square; colorbar;
    title(sprintf('context %d', ctxKeys(i)));
    xlabel('dim 1'); ylabel('dim 2');
end
sgtitle('Run-averaged state\_given\_context\_probability (reference frame)'); %[output:0067be03]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:0166c791]
%   data: {"dataType":"text","outputData":{"text":"Paths added. RealTimeCOIN on path: 1\n","truncated":false}}
%---
%[output:6eb6090c]
%   data: {"dataType":"text","outputData":{"text":"Ensemble (MD): runs = 30, motor trace = [2 90]\n","truncated":false}}
%---
%[output:485352a6]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcsAAAEVCAYAAACPNs4YAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ1sXtWZ5w8arYQR3c04yw61DTiqnLBarRw1RQkWFUkRbUYiwIhqHUe7k7iGSYOa8UASf4SykKHEXwlNATXNph6H2Y7BbREfQeqKlhLclYEpYYhmq2ESMRhIUyRIYLV0zGi6m53\/Jc\/LeU\/u93vv+55z7\/9Klu33vffcc37nOfd\/n+d8XXD27NmzigcJkAAJkAAJkEAggQsolrQOEiABEiABEggnQLGkhZAACZAACZBABAGKZQCgM2fOqL6+PnXs2DHfM0ZGRlR3dzcNLCcCwh\/JT05Oqubm5pzuxGTrTUDqdtWqVWpwcNC7\/UsvvaQ2bNhQycr09LTC93IsLCyooaEhdfjwYe+jdevWqdHRUdXU1FTv7BfifubzzeSdpJAnTpxQvb29auvWrYV+JlIsU4olLqvFwJIYYxnPtU0s8UB45JFH1F133eXcA3psbExde+21VeKTlU2l4TIzM6MeeughNTU1pTo6OpQ8bE+dOlWVLb19oQwHDhyo+n7z5s0Vsc2qPGVJx+RZ67MM6eGFp8gvthTLCLH082zQ2IeHhxUba36PFpvEUrwe17wZ3Rur9WHoV9NpuEi9Xn755RXP0GxP5v8ippdccon3MMaBqM97771XEdz8LLGYKYtYZmUXYgtFjrhRLFOIpfmQOHnypBeGWLFiReUBYD7spYHjN0JP27dvV\/ImHWWwfm\/eYpTynX5vv4ekGXbRH\/x6Xq+77jr1wAMPqM7OTu\/BdPr0aa9s+lu\/2SDk4Yay4TsceJnQzzPDbGYaZv4efvhhdfDgQS+tOG+r+gNVyoBr5YVGf5M2eZt8pewI\/Zr51iMKZp5bWlqqHt7C5c4771TPPfecF9JP8jAx3\/71a\/1eJqTejx49qvbv36++\/\/3vV8KWErocGBhQ3\/jGNzyut912W+Vvvcz4zu9hKuVBPq644oqqsGncSIvwDHvR1O+Drg6\/a5I87IPse2JiQu3YsaPKxnSG8HwXL17sCXNUuzXDxLrtpZFbvV0vX75c3XfffV4y5gtb2LNBr8e7777ba0e\/\/vWvVVdXl3rxxRcr2dLT1NuyXxmEu6SH5wLaEziZz8A05bb5GoplCrEUg5EG7ydYQWLp1wdqPqj0LAX1ncqDua2tzevLwQPSDGuJgKIh+\/W\/SiPx+x7fyYPVzLMuCmbjQt6\/+MUvql\/84hcVYfATHBFWPAzD+ofD2OicgkJ5ftUbJYa4RsqIlwW9L01EQR4OZuhQFw0\/NlEvRrje78Er5RCbq1Uso+wwL7E0hVCvH73cfg9w\/WUhLB2zzv3sS7dvnC8vZEFiGcYL\/aZ6f6p+\/yQvR3HtOazd6raLELf5wvXZz35WLVu2TB05cuQ8sXzwwQfPC3WbAm2mJ+1Ef6mI83JrsygG5Y1iGSGWQQN8dMFIKpamV4gsiNBFGZGf12g+OIL+19\/m9Yfh0qVLK2Ia1rjNe0N05CEh1+kPJnx24403Vs4RoTAf9MePH\/cESRcxKUMasZT7SBklDbDVw3fyooFBI35s5DO\/cKOk7fdQl8+efvppz8OOWwape7mfn42Jrfg9nMwHvV4+k73u5Zp1hheYKLHUPb4k4ekwjzBo0ImfMKYVyyTeue5Z6rykvUtd+L2wRrXjqO\/1lz+pO7\/Pgl42zDZg1pFZD35p6\/URlZ5pexDqoh0UyxRiGRQKiROG1ftZTAMzvRj9Pn7ehvkARD\/QPffco3bt2lXlafoNjpBi4+Fx\/fXXh\/YBBd1bRFZ\/M8ff+oNM0vZ76RAxePXVV88L2ybtszT7tRBCNfu+wkJs+tuwmZaIudQHymh68\/hM8ix1LOVK2rcdJARSj3qdhXlFYWIZVGdm2Fr3hIPCo3HFMm4fqsn\/pz\/96Xn2kUYszT7OKO9cF8uwdgthMKMIZkg+qXD4vYAjDd0GZDR+2LPB73w9HanfoPC42YaC0otbt0k52HQ+xTJCLPWHihiK2RD8HtRBgxLCHm5+Yqn3MeGBrod8zNGC8JB27typdu\/eXdV\/GlcszQeo+TAx7+2KWIo30SixTBqKSyKW+kPcFOssxFLPu\/mgTDrAx++BGvaZdC1IuwiKjOhTTPyac9CLl8kLomd+FseD172oqD6\/uA\/\/OGIpL6LSbv2eDUGevPl5UrEMm9oTp6shLgebzqNYJhBL\/Y1MD62ZYRn9TdMMAYaJpV\/ownyz8wuNIM2weWpRo3eDHiZR904bhjWRS96zCMPKiEndswwSy7hhWL8HV5IwbFKxTBKGhcduegf6y5z5UPQLueqCJXk1++X9rgt6oIc94Pwe3kFjAKQuZZBZ2tGwUWIZxjCpWErZ\/V6gkzz444RF33rrrapR+WFhU1PAag3DBomlPnYiSXldOJdimVAsdYM0B1uEDQLAbWRUXdBgAj+xDBocg\/R0g9XzZfaRRQ0SCuqcj3PvWgb4RA1UQBnj9veFhWGDxBK8g8ro11+oj15OMsAnqVjGGeATdo6fWIJl2KAtk7VfvUpTMfvc447qxvV+XnPQ4Cyz3mRBAslH3PB2kFjGYRhHLOUcv\/YveYx6YTUfQ2ED1qTd4H7m4DNJx+xjjBJL3REw86KH2YM81aTdJi6Io5lHimVCsTS9OL\/Od5xjTn1IK5amEaPxYYI5Gon5sAhrkGHTHMIMXQ\/hBt07zdQRs5\/LzF8tU0fkZcR8MAcNQgibOgL+5kPVbzATzguaOpJULMUkw6aO4ByTGYbzv\/baa1X91frLAF489OkS+hQmv5cS\/f6oL5nCECRiccoZFO4z68BMq5YVfMLsO4phHLHES5ef8PoN\/oor8LrXftNNN6lbb73VM4uggTr4zq99xg3Dis1FhZGD0ksTZXBNMCmWrtWYRfnVHxDyEPAL6VmU5dJnpdEegNxfX5SgLJUCoWlvb4+1JJxr4sNFCcpixSxnagJB4bpaRwOmzpDlF4YNtkLW44ad0xaz0WKJfMNm9OXu0pbFpetQ5rm5udjr2bomlrBrLnfnkkUyrw0hEDRRuYhzrWoFTLH8NHysL6ReK9eiXe+SWEpeuZB60ayQ5SEBEiABEiCBhATYZ5kQGE8nARIgARIoHwGKZfnqnCUmARIgARJISIBimRCYrae\/++676tJLL\/Wy9\/HHH3s\/ixYtsjW7zBcJkAAJOEWAYulUdQVnFuI4Pz\/vDU1\/\/fXX1ZVXXqkuvPDCgpSOxSABEiCBxhIovVhiL8rHH39c3XLLLQrLn7l8fPjhh96kdIwypFC6XJPMOwmQgG0ESi+WMpl2dnbWabGEZwmPEp4lQrLwLPWDYVrbmh7zQwIk4BIBiuVLL3lLx7kkln7CB6ODN4kfvz5LhmldapbMKwmQgG0EnBJLcw1S2c\/NhGqu0xi2HqOLnmVa4WOY1rbmx\/yQAAm4QsAZscQqEdjbcXx83GMrf\/utEoNVUnBgoWhZ3mv9+vW+azK6KJYoW1LhiwrTumKwzCcJxCWA8Qg88ifg+liPuIScEUtzbcUkixLr4mmCcVEs0wgfxDUsTBvXYHgeCbhAAEK5Y8cO9fLLL7uQXafzuHLlSm83m6KLpjNiaQpemADqlieeJbxMvx3VRSyx1RYq24UKp\/A5\/Wxh5utAQNo1HuKtra11uGM5b4GXke985ztOjflIW1NOiaW+vQ08TcwrhAgGHX672Qd5lvJ5f3+\/wk8RD46ILWKtskx+BFyMGLlYk2XiXGixFOODaGI399HRUdXU1FRlk+YbqCveZZqGlXZgUJp78RoSaCSBMj3Eybk+BJwSSyARTzJuGBbX6IODzAFBrjaqtF5i0oFB9TFD3oUEsiXgarsWCtJ9dOzYsSowYSP75UQ873bv3q327t2rTp8+Xfm7ubk5W8hKeXtYujb1Li0EZ8TSDLsmGeCDCsX5k5OTyjQYlypbF0iIHhYhWL58eezl7dIMDEprWLyOBBpJwKV27cfJb6yFTInr6enxHX9BsczX4pwRy7RTR8TAWlpafPs3XWlUEEosjK6v\/4qF0yGYcZe348CgfBsTU7eHgCvtOohY0MBE3UmQMkoaGKS4dOlS1dfXp+CRdnZ2qm9+85vq4YcfVitWrFAPPPCAwnNwampKZbUxu+uck1isM2KJQgUtSmC+cZmLEqxbt863vxJpulLZ0t8IgcT6r\/AoRTj9lrdLYgQ8lwSKRsCVdp1ELHUBXbx4cdVcc31qHabN6GHY3t5etXXrVm+eedj4jTQ24DrnJGV2SiyTFCzuuS5VtvQ3ilDKziJhW3IdOnRIbdq0KS4OnkcChSDg167X7H\/R6rI9v+XqSv6C+ixHRkYCF1d59NFHPafAFEt9ARdwkfPMwY5p4Lj0\/ExTPv0aiqUjHdRmfyOm0UTtWQnPc8mSJers2bO12gmvJwGnCPg9xC\/Y\/ozVZTi754bzxDJofjhOlKlxcpFE0EyxFC8T4zUolulNgGLpiFhG9Tf6jY5FuPaPv7pG\/eWPn1erV69ObyW8kgQcI+C6xxNnMRV90KIughTLfIyVYumIWJrVb4ojxBQ\/+ubPW7ZsUTcsPqR+8MbN6oknnvC27tIHCcEzlV1K8DkPEigKgTKJJcKpQ0NDXtX5hWHpWWZj1RRLR8XSb4EBfCabP0MYN3xliXp8m1K37FXqL576OyWDg9DXKdNOEKqVvs9sTIqpkEDjCRRdLPVBjBjhunPnTvWjH\/3Im1uJAyNicchoWHzOMGxtdkmxdFQsUe36AgP4H0IJQYRoQiyf2btGbVun1A9fVOqKm57wvEoRSvkdd9pJbWbGq0mgvgRcF8v60kp\/tzJxplg6KpbmgB8Jr0Is8TM8PKyu+b19qmvpJw3h9p\/crPbv319ZwEA8S33aSdpVgdI3NV5JAvkQKNNDPB+C8VItE2eKpYNiCVFDXyP6JPG3TB3BZ3L8q39+V838yXzl\/5U7lWrtWOWdK4ecj35ODCfn2rHxHhA8y34CZXqIN7I2ysSZYumgWIqooZFg6avr2l7zwq1Jj72Hlbp4xT3q3nvvrVzKtWOTUuT5NhLI+yEuC4LYWPZ65ilvzvUsS9S9KJYOiKVfeBQViz5K9D0++eST6i+\/vcUbzBPneOe0Un99ql29v3hjlVBy7dg49HiOCwTyfoivWbNGPf\/88y6gyDWPeXPONfMJE6dYOiCWZngUYVNzqTv8f\/f2XnVz+5FKP6WfLUAotz\/e7s29RDr6ETSXk32ZCVsVT284gTwf4rLYx5tvvnleG2p4weucgTw517kokbejWDoglqhFPTwq8yNljqSs5INz7rjjDm9upQzs0S1g7rhSf3txddg10kKUYl9mHEg8xyoCeT7E0W3x0dFd53VhZAnAXJ1H0sbi6H67J2V1byyGsG3bNm8qSpzF1vPknFWZskqHYumAWPqNfMWIVxzmurBoyP\/xo12B3uXOFzd5uw4kPdiXmZQYz28kgTwf4v\/lq2vU+JePqIFnV6v\/\/uN8Q7FRK\/lkzZhiGUyUYumAWJrhUb\/VemRkqzRkqXKEXS9b\/KkBYFTs2+9HrxVrhl4x1QRhW1kFKEiss268TI8E0hDISiyxEcELL7xQlYV\/\/PtDat8mpf7skFIXLavepOCKK66oGgeQJu\/6NX5i6bc1F+ZL43N4pO+99563JRdW83n66ae9aWRYuABLXn7mM5\/xtio0d2bC9l7wWrES0OHDhytbeWHzaGzujMPPq82Kc62c6nE9xdJSsfTrJ4QXKSLl5+mhL+XhrUu8kbEQydl\/uFQd\/Wi1+n9vPeY1bhxo4GPT0X0tQdNI8HmYWNfDaHkPEogikOVDXMKuQSPO0dZ+OHf+yPKoPMb53hRLfV9fhEn1rbmwh+X27dsr+1Xq52JLL6zqA1GFWEJUceBvsJLrcJ6EYfW\/5V64Blt9yZEl5zg8GnkOxdJSsfQTKxiKvvmzeHoYEYtDQrDwJDGIZ+e39nvzMSGwf\/Kf\/1D1r\/5k3uUzp+OFYsNCrwzLNrLZ8t5RBLJ+iB85csTblODH26ojNRDKfUfa1dKrq0eWR+Uv7vdRYVh9AXWIpb64OoQUzwsIIg75\/7bbbvOEU3Y00fcDxubRQWLpl+esOcfl0ojzKJaWiiWMwU+QzM\/0PkuEYP\/gn454Aw8QTtFDpxBNbAK74uIj3vJ3UaHYsGkknGLSiKbKeyYhkMdDHIIpS0hKXhCp2bI7v119\/MQyaGsuiKW+V6XuPfqJJc7XD+yVef3111cN8IF3iufGqVOnlGwBpu+DmQfnJPVcz3MplpaKpZ8ghYkU3iDx5vulGz55ww2aBoI+GBh\/1LD3sC3BorYLq6cB814kUC+PB23nM\/+r1xsD8OJxpf7T1Z+su\/xvrp3KbYN1UyylX1JGxJqepS6WYZ5l0IjXsAE+ZnrgTrEsUfuztbL9BAnVAg\/RnDKCz7\/3ve+ptWvXeoNwzBGyZnVCWHGY8yxLVO0sasEJ5NGuEbnZuuKI18Xx53um1H\/d3ut1bbymNqn79iQfYR6nCsLE0tyay\/Qs4\/ZZive4Z88epYdhkT99ey+IJQ72WcapuQKek0ejagQmruvaCOq8p60Esm7XshDBPfd8Ok8ZnyFK88bfHlGzr0QPmkvDyhTLsK25jh8\/XhWGldArRsNiJOt1112nfvvb3\/qOhkUIFiIo6R89etQbKPTqq696o2lxMAx79mz0PII0tZzDNXizkYqTyvW7jRiYxOT9Klmuy7pR5VDs2EkGDbrhCjyxEfLEghDIul2jvxI\/+jrKggrhWRybNlVPI7EJpYhgV1dXlWdYax6z5lxrfvK83pk+Sz2kACADAwNqfHz8vFUmTKOQ\/zHPSEaF6UCLUtlRA3L0UbTc7DnPJsW0bSBQlHZdC0tzPubmzZt9n4FZ3GN2dla1tbXVkpT11zojlvp8IsTqMdILfW56\/DyItnltEcUyatANp3pY3xaZwQwJUCwzhBmSVJk4OyOW5jBo8\/8w0yiDWIaVvx5TPebP\/KM69MpJ9cIbp9XzW66uT0vlXUgggIA8xLEyTdE9nkYawcmTJ70VfuhZNrIWjHubnqTfMGa\/7Er\/5fr16329ULNRFbFhRXmdtVbzmv0vqiNvnK4kA7Fc\/Tltjb1ab8DrSSAhATzEd+zYoV5++eWEV\/L0pARWrlzpDSwq+uGUZ6mHXeOIpfRXohKxTqI+mVYq1ozr9\/f3K\/zwCCcAT7J35pjC7\/kPFlT77zep9uaLPNGkWNJ6bCAAwcQPj3wJwMEoopNhUnNKLJF5GaQTFYaNI5RIT8RyYmJCtba2epVehoqvpfnc++zfq0d+edITSXiQEMize25Q9z57XO169jjFsha4vJYESMBKAs6IpelJhg3wiRoBq9dEmTqos7LAC7Y\/U5UUvMo377quIpb3fHmpuvdffniQAAmQQFEIOCOWcaeOoGIgpFjLMCj0SrGszXwP\/fIdLwQrhx521YUUIrrxqssonLXh5tUkQAIWEHBGLMEqaFECc9V8rKhvLhIctMM4Pcv0VihhV6SAcCxEE32YCMViZKwc4nmmvxOvJAESIIHGEnBKLPNARbGsjaoM9EG\/pXiSm77Qppbs\/nnlf4Zka2PMq0mABBpPgGJp6a4jYaZh4\/J1ZmhWz7+MlOX8y8Y3eOaABEggHQGKpYNiaeui6bqXaZojQ7HpGiivIgESsIMAxdJBsYTp2Lx8nblIAfLLuZd2NHjmggRIIB0BiqWDYlmP5evSmdOnV2FUrMzBxKf0LGslyutJgAQaSYBi6aBY5r18XZYGiYE\/vY+9VlnlZ2r98qql8PTQ7Zs7v+StAsSDBEiABGwjQLF0UCxtM6I4+dFDs3dd16G+9YfLvGkmmK8p68piFC3ElAcJkAAJ2EaAYkmxrJtNmn2ZCM1iyTys+IPdSmT6iel91i2DvBEJkAAJBBCgWFIs69Y44Eli\/qV+6H2Z+iIH9DLrVi28EQmQQAwCFEuKZQwzyfYU8TDFs8Rv8Sb9FjngogbZ8mdqJEACyQlQLCmWya0mwyv8lsxD8kGfZ3hrJkUCJEACsQlQLCmWsY0lrxODvEn9c1l7Nq88MF0SIAESCCNAsaRYWtNC9CXzII5T3Z3eVBIJ21IwrakqZoQESkeAYkmxtMrozekksjcmBdOqamJmSKB0BCiWFEsrjR7TSCCQODAACMvlyZxMephWVhkzRQKFJkCxpFhaa+DwMrEvJvbHxAEv861zn1Ewra02ZowECkmAYkmxtN6wzSXzkGEsZsD1Zq2vOmaQBApDgGJJsXTGmPXpJJJpfY6mMwVhRkmABJwjQLGkWDpltAjNoi8TnqV+yEAgpwrDzJIACThDgGJJsXTGWPWM0st0stqYaRJwlgDFkmLprPHSy3S26phxEnCOgFNiOTMzo4aHhz3IIyMjqru7OxL42NiYam9vDzz3JYplJEPbTzB3M0F+2Zdpe60xfyTgFgFnxPLEiRNqYGBAjY+Pe4Tl746OjkDiEMoDBw6ECivF0i2DDcqtX1gW57Ivsxj1y1KQQKMJOCOW8Crn5ubU6OioampqUmEe45kzZ1RfX59atGiRx3ft2rX0LBttaXW4v7mQAW4pU0y4R2YdKoC3IIECE3BGLCGOOAYHB73f5v96HUEsP\/jgA9XS0qKGhoZUV1dXpFhOT0+rtrY274eHuwTMRdmxeAEWNsCBVYDwPw8SIAESSErAKbHU+x7hac7Pz1fE06\/gCwsLscVSru\/v71f44eE2AT0si1AsVgFiP6bbdcrck0AjCVAszw3wmZiYUK2trfQuG2mNGd9bD8vCo8T\/ss4sdjPhQQIkQAJxCTgllnHDsFL4JJ7l7OwsQ7Bxrcah88xdTJB1LpPnUAUyqyRgCQFnxNIMu0ZNCQFfiqUlVmZBNszpJRRMCyqFWSABhwg4I5Zppo5QLB2yxDpkVd9cGrfjziV1gM5bkEBBCDgjluAdtCiBiGJPT49atWpVpWoolgWx0gyLgbDskt0\/r6RIwcwQLpMigQITcEos86gHLkqQB1W70zT7MSmYdtcXc0cCNhCgWHK5OxvssCF50KeXbPpCm8LCBTxIgARIwI8AxZJiWeqWoU8voWCW2hRYeBIIJUCxpFiWvomYq\/5gpR\/Owyy9WRAACVQRoFhSLNkksIastqk0V\/qhSZAACZgEKJYUS7aKcwTMkbLcsYSmQQIkIAQolhRLtgaNAKeW0BxIgAQ4wMeHAKeOsGGYBPRBP\/iOYVnaCAmQAD1LepZsBT4EIJi9j73m7YcpB8OyNBUSKC8BiiXFsrzWH6Pk+lxMnM7pJTGg8RQSKCABiiXFsoBmnW2R\/MKynF6SLWOmRgK2E6BYUixtt1Er8ue31RcEE0vl8SABEig+AYolxbL4Vp5hCc2dS9iPmSFcJkUCFhOgWFIsLTZPO7PGhdjtrBfmigTyJECxpFjmaV+FTlsf\/MPpJYWuahaOBBTFkmLJZlADAdPLZFi2Bpi8lAQsJkCxpFhabJ7uZE33Mrk\/pjv1xpySQFwCmYvl2NiYOnDggHf\/zs5ONTk5qZqbmyv5wYo5OMf8PG6Gsz6PK\/hkTbS86ekLGTAsW147YMmLSSBTsYQInjp1So2OjqqmpiY1MzOjhoeH1fT0tFq1apVHkGJZTENiqT4loHuZDMvSMkigGAQyE8szZ86ovr4+NTg4WBFGEccNGzaokZER1d3dTbEsht2wFBEE9IUMGJaluZCA+wRyF0tdMOFh4qhHGFa8WtxPhNqvuhiGdd+IbS2Buan01PrlXMTA1spivkgggkBmYikiqIdh9XuLKG3evNnzLvPsszxx4oQaGBhQ4+PjXhbk746OjvNwUCzZRvImgAXZD71y0rsNvcy8aTN9EsiHQKZiKYJ5+PBhNTU1pUxxWlhYUENDQ+rtt9\/OVSzhVc7NzVX6TuHJtre3e2Fg86BY5mNYTLWagL7yDwf\/0DpIwD0CmYulDQggjjjQfyoCrv\/v5\/EiRNzW1ub98CCBPAggLLtm\/4uVbb84+CcPykyTBPIhUFix1D1JeJrz8\/MV8fQTS\/msv79f4YcHCeRBwFzEAF4mdzDJgzTTJIFsCVAszy1KMDExoVpbW+ldZmtfTC2AgLlP5lR3p9p01WXkRQIkYCmBwopl0jDs7OwsQ7CWGmlRs2Xuk8mNpYta0yxXEQgUUizNsCsH+BTBVItZBr+wLKeYFLOuWSq3CeQmlpi+0dvb663oYx5+y+BliZFTR7KkybTqQcAMy3LwTz2o8x4kEJ9ALmIpU0RaWlp8B9XEz176M7koQXp2vLIxBMywLOdkNqYeeFcS8COQi1gGLX1nYxVwnqWNtVLePIWFZfFde\/NF5YXDkpNAAwnkIpbiWfb09FStE9vAcgbemmJpY60wT5iPCU9TDoRldz17XDE8S9sggcYQyEUsURTbdhcJwkuxbIzh8a7RBPRVf\/SzEZ7FVBN6mdEMeQYJZEUgN7Fs5ACfJHAolklo8dx6EzD7MfX708usd23wfmUmkItYShi2q6vLdz1Wm4BTLG2qDebFj4DZj2l6mVgBiAcJkEC+BHIRSw7wybfSmHo5CZjTS4QClszb+C+r\/9z75aXlBMNSk0AdCOQilhzgU4ea4y1KSSAsLMupJqU0CRa6TgRyEUvkHX2Wu3fvVnv37lXNzc11Kk7y2zAMm5wZr2gsgbCwLL3MxtYN715cArmIpYRhjx075ksu7xV8klQXxTIJLZ5rEwF9U2kzX\/Qybaop5qUIBHIRS5fAUCxdqi3m1SQQNL0E59HLpL2QQHYEKJbntujiriPZGRVTqi8Bc1Npepn15c+7lYMAxZJiWQ5LL3gpw\/oxxcvkbiYFNwIWL1cCmYmlPl1k6dKlqq+vT7HPMte6Y+IkcB6BoOklciL7Mmk0JJCOQGZime72jb+KfZaNrwPmIFsCftNL0H+J5fHwHf6ml5ktc6ZWfAK5iKUIkIlvenrauoXVKZbFN\/IyltAvLKsLJpjQyyyjZbDMaQlkKpayGMHbb7+tJicnq+ZXSpj28ssBUSBDAAAYSUlEQVQvV6Ojo6qpqSltnjO9jmKZKU4mZhkBv7Dspi+0qfkPFuhlWlZXzI7dBDIVS2y4PDc3FyiGNq4ZS7G020CZu9oJ+IVl4VVe+7nF3rZf9DJrZ8wUik8gM7GMu8RdlKDWGznFst7Eeb9GEAgKy2LnkkdeOUkvsxGVwns6RSAzsYy7eLpt+1xSLJ2yV2a2RgLmptJIDoKJQ7xMhGkxAIgHCZDApwQolpxnyfZQMgJ+\/ZgyQhaCyRGzJTMIFjcWAafEEiHc4eFhr2AjIyOx9socGxtT7e3tgefSs4xlJzypYASCppfAo8R39DILVuEsTs0EnBFL7GIyMDCgxsfHvULL3x0dHYEQIJQHDhwIFVaKZc02xAQcJRC06g\/CsAjN9s4co5fpaN0y29kTyFwsg1bt0bOeZtcRc2BQmMco\/aeLFi3ybrt27Vp6ltnbDlMsCIGwsCy9zIJUMotRM4HMxLLmnEQkAHHEMTg46P02\/9cvh1h+8MEHqqWlRQ0NDamuri6KZd4VxPSdJhC0qTQ8THia9DKdrl5mPgMCToml3vcIT3N+fr4inn4s4szrlDAsVhdqa2vzfniQQBkJBIVlZaUf3QOFiN57bhRtGVmxzOUjQLE8NxpWqr6\/v1\/hhwcJlJWA36bSMloWv+llltUyyl1uK8VSH\/Uq\/ZsHDx6MHYaVKk3iWU5MTKjW1lZ6l+VuDyz9OQJBm0qLR0kvk6ZSNgJWiqVfJZhh16gpIUgjiVhy8+eymT7LG0UgaFNpCcvqYVvuZBJFk9+7TsAZsUwzdYRi6bp5Mv+NJhDUj6mLI73MRtcS718PAs6IJWAELUoQtC4txbIeJsR7lIGALogQSuxagkPCsvQyy2AF5S6jU2KZR1VxUYI8qDLNIhLQp5cgFAuBhGjq+2LSyyxizbNMIECx5NqwbAkkEJuA6UHiQgimHpbV+zrZlxkbLU+0nADFkmJpuYkyezYSEA8SYgjP8tArJ6vCsviHXqaNNcc8pSVAsaRYprUdXldyAnpYFqv8wMOUHUue33K1am++yAvVYlsw0\/ssOToW30ECFEuKpYNmyyzbQkAPy8LDvPZziys7lkAw8Rm9TFtqi\/mohQDFkmJZi\/3wWhLwCMim0gjLbrzqMt8tvkwvU7xPIiQBFwhQLCmWLtgp8+gAAb2P0gzLYp9MepkOVCKzGEiAYkmxZPMggcwImNNL9LCsvvi6eR68TB4kYDMBiiXF0mb7ZN4cJGD2Y0IkEabFoc\/J5EIGDlZuibNMsaRYltj8WfQ8Cej9mPAcg3Yr4RSTPGuBaWdFgGJJsczKlpgOCZxHQLb7ksUJEH7d9exx7zw9LEsvk8ZjOwGKJcXSdhtl\/hwnoC9ggIE+OPzCsvicXqbjlV3g7FMsKZYFNm8WzRYC+v6Y3pSRkE2k4X3CI5WFDDjFxJZaLHc+KJYUy3K3AJa+bgT0EbCyYEGYJ0kvs25VwxvFIECxpFjGMBOeQgLZENAFc6q7U2266rKqJfH00bK4I6eYZMOdqdROgGJJsazdipgCCSQggME8S3b\/3LtCH+Sjj57VFzHg4J8EcHlqbgQolhTL3IyLCZNAEAFdALHajwz80fs2dSFFOnpYduKGf6+2r\/4cAZNA3QhQLCmWdTM23ogEdALm4gWyio\/pSeoDfPTvTDElXRLIkwDFkmKZp30xbRKIJCDhV7O\/Uvck9R1MkKBcY34eeTOeQAIpCVAsKZYpTYeXkUB2BIIE09wzU8K1er8nBTO7emBKwQQolhRLtg8SsIKAvtrPm3ddV8lT0AAfEVLM2dTPt6IwzEThCDglljMzM2p4eNirhJGREdXd3e1bIWfOnFF9fX3q2LFj3vfr1q1To6Ojqqmp6bzzX6JYFs6oWSB3Ceir\/WBfTAz+aW++yCuQ37xL+cwM4bpLgDm3lYAzYnnixAk1MDCgxsfHPZbyd0dHRxXbhYUFNTQ0pLq6ujwxlf9bWlrU4OAgxdJWS2S+SOAcAT30io\/gOWIwD+Zk+s27lBAuB\/zQhPIk4IxYwqucm5ureIhjY2Oqvb090LvUoZnX6t\/Rs8zTvJg2CaQngGkkj7xy0hNIEU3xNs0dTGR5PPZfpufNK8MJOCOWEEcc4h2a\/4cVM45YTk9Pq7a2Nu+HBwmQgD0E0Gd56JWTld1KkDN4kW+d+xz\/Iwwrovrmzi9VQrf2lII5cZ2AU2Kpe5IQwPn5ed\/Qql4p0n+5fv16Xy9UPEu5pr+\/X+GHBwmQgF0ERDQf+eU73iLrQQcH\/NhVb0XJTaHFUvorUVlRA3wmJiZUa2srvcuiWDbLUWgCft6mXmAKZqGrvyGFs1Is9VGvnZ2danJyUh08eDBRGDaOUCJB9lk2xO54UxLIhECYaHKEbCaImcg5AlaKpV\/tmGHXsAE+USNg9fQplmwLJOA+AVlTFgIJAZUwbZBgisii5Pr0FPdJsAR5EXBGLONOHQEoCOmpU6cCQ68Uy7zMiemSQOMILLn\/OU8kz+65wds8GoOC5EBYFvM1r\/3c4qqBQfL9n39lqbr7+qWNyzzvbD0BZ8QSJIMWJRBPsqenRy1durRqQQKpAQnnNjc3V1UKPUvrbZQZJIFYBEQgZZ9MfQcTvwTgdUI8dz17vGqrsFg340mlI+CUWOZROxTLPKgyTRJoDIELtj\/jTSPBfEuEWiGEuocZlCsR2Hrk+t1331WXXnqpd6uPP\/7Y+1m0aFE9bs171ECAYsnl7mowH15KAnYR0DeQln5LhGAhoGGiue+m\/6A+WPhnde+X8w\/FQhwx7Q1T4V5\/\/XV15ZVXqgsvvNAukMzNeQQolhRLNgsSKAwBCb2KQGLFHwgljr4fHlN\/8dfveH9f\/vtN6u0PFqoWMxAI9Zh28uGHH6rXXntNrVq1ikLpiPVRLCmWjpgqs0kCtRPQdzBBahKy1QcE5S2W8CzhUcKzREgWniUP+wlQLCmW9lspc0gCGRPQdzCRpCGcG7\/Q5i3YnucBrxJhV\/ywzzJP0tmmTbGkWGZrUUyNBBwhYO5ugvVmIZiyKDvXmHWkIuuUTYolxbJOpsbbkIB9BMz5mHoOuYOJffXVyBxRLCmWjbQ\/3psE6kYA00pwoE8SA39wYGpJ2LHs312sXh9YnVkeOW0kM5R1T4hiSbGsu9HxhiTQCALop4zasUTPF5bBm1q\/PNOsctpIpjjrmhjFkmJZV4PjzUjABgKy9yXmZQYdWDYvj4PTRvKgmn+aFEuKZf5WxjuQgKUE9NAs+ihlk+k400fShFRxDX5k2ghW8tFX70mTpqVoC5ctiiXFsnBGzQKRQFoCmIe5ZPfPK\/Mvw9JJE1KFVyliKXMtdbFMk2basvK6ZAQolhTLZBbDs0mgwARk0QIssB5n6bs0IdWoa6K+LzB+q4tGsaRYWm2gzBwJNIKAGQ7FWq6y0o4sJIBFBZKuxGOu3gOvUl9UXfc8ubpPI2o++J4US4qlXRbJ3JCABQTMcKj0MeqLn+Mcv5V4wvod9+3bp77+9a9XVu+BOOJH0pU+TK7uY4ERGFmgWFIs7bNK5ogELCBghkPjhkeD+h3hnS5ZskT95je\/qfImIa74jouqW1DpIVmgWFIs7bZQ5o4EGkDADJeK5xd38XOInwggso\/0sMvIH391jfpvP\/iJ50kiBItzcOB\/\/fwGFJm3jCBAsaRYspGQAAkYBMzFziW0Gic8KkKLJOV89Hdu2bJF3bD4kHrm9Cb17W9\/2xNPCCZ+JBQroV1uBm2fSVIsKZb2WSVzRAI5EchyHmNQWiK04k0uX77cE8MNX1miHt+m1C17lfruzN9UvEl8h75KeJcQUJwfR5RzQsRkAwhQLCmWbBwkUBoCaeYxBoliWFpmGBdpPLN3jdq2TqkfvqjUsu6fqNWrV3ujaSGU+B6iCaHU52HCI4Vw8mg8AafEcmZmRg0PD3vURkZGVHd3ty\/BhYUFNTQ0pA4fPux9v3nzZjU4OOh77ksUy8ZbIXNAAnUkEHegjmQpTBSD0tLDuPj7jjvu8EKwXUs\/SfX2n9yspqamKkIooileJjxMDvipo1HEuJUzYnnixAk1MDCgxsfHvWLJ3x0dHecVc2xszPsMAnnmzBnV19en1q9f7yuuFMsYVsJTSKAgBEyPT+ZORhXPTxTNtPDckQE7eno47\/\/+n3n11J++W\/l45U6lWjtWVcKt+EJCr\/A0IaT6PEtuEh1VQ\/l\/74xYwqucm5tTo6OjqqmpScEwEeMP8i51dLp4mkgplvkbGe9AArYQMAfuQISiBtMECayZFrxDeJArLj7ihVuTHnsPK\/VR62YvKoZnm3i0yJ8IJ0OySalmd74zYmkKXpgA6njEs4SXibBGkFhOT0+rtrY274cHCZAACQiBKIE1+zTxQv\/so7u8wTxxjndOKzX7D5eqX\/1urdq\/f793CTxUGfAD0WRINg7JfM9xSix1TxKeJgwqqC8S2CCoBw4cUOvWrat4pEFiKZ\/39\/cr\/PAgARIgAZOA32AfeHsibrL8HSJWf\/X9MXVz+5FKP6UfTQjlN\/7qUrXzW\/u9AT\/iQco8TVyDUDH+x+Af0xPOcnQvazucQKHFUooO0Tx16pSvYEoYdmJiQrW2ttK7ZIshARIIJBA02CdotR+8rH+19TFfwZw7rrw5l\/LCD1FE+rI+rO7Z6oKph2LTjO5l9aYjYKVY6qNeOzs71eTkpDp48KBXQjGsuGFYXKMPDjIHBLHPMp3h8CoSKCsBUxjDVvvZtWuXuub39gV6l197Yq164oknPJQS7sXfMtgHo2L11X38+iyTju4ta73VWm4rxdKvUGbYNckAHwgizofoNjc3VyVPsazVhHg9CZSHgN9gHxE5+S0LCyBEevf2XvXgjS9VACHsetniT3lhVOyz\/\/PvKgupy7xKuQ9GxiK0K\/MvzdG7aUf3lqfGsiupM2KZduqIzLlsaWnx7d+kWGZnTEyJBIpOIGywj3h4EDQIHA5ZiEAG8Tx3crn61\/\/7f6h9mz4h9WeHlBr+7t94XqU+iEdfBUiYSt+ovlVYkmX4il43eZfPGbEEiKBFCUQQe3p6PIMzFyWIM8BndnaWI2HztjamTwIFJgCBE08PnqCEYOFJYhDPhlsHKx7id\/cMq\/7Vnyyi\/oM3PlmgIM7+leyjbJwBOSWWeWCiZ5kHVaZJAuUjYIZE7\/\/mFvUH\/3REnf63m9TGjRsrQCCk0jWEOZlY\/u7t9896Qhtn3if7KBtjWxRLLnfXGMvjXUmgYAQwvQN9jAiXQgz\/9NY\/Up+\/5iZvs2c9fIsBOzKA58knn1S9vb3qzTff9Potow72UUYRyu97iiXFMj\/rYsok4DCBpHMY9RDpvn37vJAruoXQf4nv9N1E9FWD9D0tdVx+98f3EFruSlJ\/w6JYUizrb3W8Iwk4QCBN\/6C+6bNs0QVhw6CcOH2SOpY093cAq7NZpFhSLJ01XmacBPImkKR\/UEKk4v3hf4RlkYYsOBCnT1IvU5L7582i7OlTLCmWZW8DLD8J+BJI2j8ofZbiUaIPUhZETyqSyFDS+7Ma8yVAsaRY5mthTJ0EHCXgN6dSvEURM10EJWyK63CgX9JcRCBJP2jUAu6OYnU22xRLiqWzxsuMk0C9CUT1I0IMMaBHdjgyPcqo6+tdHt4vPgGKJcUyvrXwTBIgAa8PEkvayYo74i1K+BX9lPg7aGNp9kO6aUQUS4qlm5bLXJNAAwj49SOKt4iwq2ylJWFac2Np9kM2oNIyuiXFkmKZkSkxGRIoPoGgfsS43iL7Id21EYolxdJd62XOSaBBBMyBOrLpc9K5lA3KPm+bggDFkmKZwmx4CQmUm0DQQJ2467uWm56bpadYUizdtFzmmgQaTCBu6LXB2eTtMyJAsaRYZmRKTIYEykMgy4E6SeZeloewfSWlWFIs7bNK5ogELCeQ5UAdzr20vLLPZY9iSbF0w1KZSxIoMAGGdO2vXIolxdJ+K2UOSaDABLIM6RYYU8OLRrGkWDbcCJkBEigzgSxDumXmmHfZKZaWi+XJkyfV448\/rm655RbV1taWtz1YkT7LXPx6Zh0Xv47xMClSPVMsLRfLlyzPXx7qyjIX\/0Fahjo2R7keOXJE3X777Wp2drY0L75FqmeK5Tkxmp6ettKA8Wa2YcMGZWv+8hBLlrn4YlmGOkZf5O9+9ztvT8uf\/exn6uKLL1Zf+9rXStmWi\/CCUHqxRKPdsWOHevnll\/N47jNNEiCBEhNYtmyZGhwcVAMDA+r9998vJYmVK1eqRx991Pmyl14sJa4O0eRBAiRAAlkRgGf50UcfqWuuuUa98sornmdZxgNjLYow3oJiWUbrZZlJgARyJ8BRrrkjrusNKJZ1xc2bkQAJkAAJuEiAYulirTHPJEACJEACdSVAsawrbt6MBEiABEjARQIUSxdrjXkmARIgARKoKwGKZV1x82YkQAIkQAIuEqBYWlZrZ86cUX19ferYsWOqs7NTTU5Oqubm5shcjo2NeedgTpdLR5LyzszMqOHh4UrxsFDDqlWrXCqub15RdwcOHPC+K0qZpKAnTpxQvb296tSpU2rdunVqdHRUNTU1RXJoaWlRU1NTqqOjw6n6TVJeKdjCwoIaGhpSXV1dqru726nyIrNJyqy3d9fsnWJpmWnqohdXAGVJqc2bNzsnlnHLizLiXHl5wP\/bt2938oGqm5xeruPHj1eV0TLTTJwdXQRuvPHGUEHAi9Dc3FxFTPH\/Y489FvtlMXHmcrggSXn128tL4MjIiHNimaTMci5ehPBSD5HFYg3j4+NOvBRRLHNoNGmTlLcuGBI8JhjT7t271d69ewO9Sxjg\/fffr371q19517jkWaYpr7A1r03LvNHX6S8L8jDp6ekphMdsPgzxYoCVXMK8S90jdelBKh6Wnuc45YUdb9u2TWFO5vr1650TyyR1HOd51uj2GHZ\/iqVFtWMaXpw3L7yV4pifn\/d+uySWacpbJLE0w2+uh+PMpuQXDdCjA2FNL47tW9R0vaykKS94XHXVVeqpp55yMgybpMxm9MC2+ovKD8UyilAdvzffvOStc+fOnb5hCny\/a9cudc8996iDBw86KZa65xxVXjN05VqYzjQlP08SD08svO1i35WfWOqeZBLPAhzQzxnHC61jEw29lelJRpUX3z\/yyCPqzjvv9Nqxi32WScoMsZSXehf76CmWtrS0cx3lScQDD5Rrr73WC9nF7d+0qLjnhZnjiqX00bo+GIZi6W+NeKg+9NBDzvVHJxEO6T7ZuHGjt26qqwN8kpRZ+mal3bo27oBi2SD1MEeFoXP\/85\/\/fFWHd1goSt5K77rrLm90oe1iWWt5pZqKIpQoD8Ow5zc+V4UyaRgWdvzCCy943SYuh99rCcO6Vm6KZYPE0u+2pmcVFsYxp1FIelHD8y0qrkpSXnkYFWEErF4Heti1iAN89EhJ1IAXF0fA6nVpttew8urThfQ0XBvRnqTMJg\/X7J1iaZN6KFXlISbxFpOca1OR404dkblce\/bsKcRIUd1TlkEvZZ464lpIzq8NJZlGoV\/vmocVlPeo6UHmCHbTK7XpueSXF4qlZTUUNklfOsj9Rry6KpZxyxvkSbs4N800ubIuSqDbc5Cn5Vq\/dNgE\/aD267JYwpaTlFlv764tPEGxtEwsmR0SIAESIAH7CFAs7asT5ogESIAESMAyAhRLyyqE2SEBEiABErCPAMXSvjphjkiABEiABCwjQLG0rEKYHRIgARIgAfsIUCztqxPmiARIgARIwDICFEvLKoTZIQESIAESsI8AxdK+OmGOCkrAXPJPL6Y5nzDPHRrC5usWFD2LRQI1E6BY1oyQCZBAPAJBe3DKpO6tW7fmvtuILO7g2rJq8QjzLBLIjwDFMj+2TJkEqgiEbVid93JvskrM0aNH1SWXXOLcRuE0JRJoNAGKZaNrgPcvDYEwsTSXPNPDsCdPnvR2o8Ham\/fdd5\/Hq7OzU01OTnr7mMregGHeor6I9YMPPuil4dJG4aUxEhbUWgIUS2urhhkrGoEwsURZ9fV9TbHs7e1VK1as8DZDxoH9Dw8fPqykrzPJQvOuriNcNHtgedwiQLF0q76YW4cJ1CqW+o4r5gCgqLR1bBRLh42IWW8YAYplw9DzxmUjECVoYZ4lwrDj4+Oqo6PDw0axLJv1sLyNJkCxbHQN8P6lIRCnz7Knp8cbfOPXZ0mxLI2psKAWEqBYWlgpzFIxCUSNhpVNoJubmymWxTQBlsphAhRLhyuPWXeLQJJ5lvQs3apb5rb4BCiWxa9jltASAmlX8JGpIwzDWlKRzEYpCVAsS1ntLDQJkAAJkEASAhTLJLR4LgmQAAmQQCkJ\/H9vtX3OlBUTPgAAAABJRU5ErkJggg==","height":277,"width":459}}
%---
%[output:7c560aef]
%   data: {"dataType":"text","outputData":{"text":"Run-averaged responsibilities: [1.00 0.00 0.00 0.00 0.00] (sum 1.000)\n","truncated":false}}
%---
%[output:415ebfed]
%   data: {"dataType":"text","outputData":{"text":"Pooled state mean = [ 0.325 -0.260]\n","truncated":false}}
%---
%[output:6df3ce89]
%   data: {"dataType":"text","outputData":{"text":"Pooled state covariance:\n    0.0031   -0.0023\n   -0.0023    0.0021\n\n","truncated":false}}
%---
%[output:0067be03]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcsAAAEVCAYAAACPNs4YAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnX+UFtWZ559uGhBRVKD9Aaygs7gze3aGXbMzy5I\/cJd42GEOZCZn5iCdBGyxh0xEzQYEoVUkhu4WaI0NYYYlDJJM8Gc0gZnMECTYZgRBUVvDD0EUtG2jBlAQmh\/9Y88tzn29fal6q5768b5PvfV9z8khwq2qpz7P873furduVZV1d3d3E34gAAIgAAIgAAKeBMpglqgOEAABEAABEMhPAGaJCgEBEAABEAABHwIwS5QICIAACIAACMAsUQMgAAIgAAIgEI0ARpbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgltH4YWsQAAEQAIEMEIBZZiDJOEUQAAEQAIFoBGCW0fhhaxAAARAAgQwQgFlmIMk4RRAAARAAgWgEYJbR+GFrEAABEACBDBCAWWYgyThFEAABEACBaARgliH5tbe309133+1s3dDQQP369Qu5p\/M3e\/DBB2ns2LE0evRo1j5feuklam5uprlz57K2i6vxkSNHaPr06XTTTTfR5MmT49ptqveTZE7C1knSQFVcbW1tsegiyL4U49mzZ9OaNWto5MiR9MQTT9Djjz9Oq1evpoEDB5733\/v376e1a9dSbW1trLrVXLUOlA65GrZzo\/uZDRs2OP+0bt26yPtMOv9J7F8zbWlpoSFDhuRyncSxvPYJswxJOymzVEKurq6mpUuXskSh41GFVCyzDImyZDdLMidh66QQsIMYXNA4wuzLNkv7WGH2GTRe1U7tX\/3i0KF9IcCJo5Ta+uW0EOcKswxJGWYZElyGNoNZRp9xCWNsfh1rmH0GLVtlbmr\/elQbdDuvdn7nEnX\/adk+yZwFZVB2zTXXdKvGnKGtSuC8efNyxzC31Ve8t99+e4+pOD2MVtMS+opLFVZVVVVuPxMnTuwxdaMLb9y4cfTQQw\/1mIbQx1HTPfpXX1\/vekw1dFe\/GTNmOH\/aU0R+cahtzGkA9d8q7uPHjzv740zDusWtp1byxZHvfO3YzHzY0zgq3rBTOXYM9957L73++utO7aicmtOw119\/vTNCjqsO1DSbGm0r0Zj5DHP1bjNxq32\/mtCdmDq+ik3Xoa7BKDnR+jLrWbP\/0pe+RF\/72tcctvpn6yaf+PV+5s+fT6tWrXJlqfmoKc3Nmzc7bcxj2PpXujLzoDu2iy++2Km1fP2Larty5cpcyKNGjephNEH2FXQadvny5bR48WLSU5paC+rWhdqHbXDcDtrt4kjXwaRJk5z9qzrRvPJp0+3fguYgav78atusL7sW\/Ppgjs+49bu6Hx8xYgRt3bqVdI2Z+83nT2p7fX41NTU0c+bM3OmoWh00aJCjLVvPqpEzstRw33vvPd8rIrcrHVVUutjUvTu3e3luBa0MV3fcbiM13WHZnYHbFJRuq\/en26httZC1MN2KLl8cQffld4XidiFhc3E7tyDnm0+oV199dc7QbU5+Met\/19tpMZhi1uI3zVJ1DnHXgdmR2vEEPQ8do8lE1fSyZct63PNyq01TH1qQQWrJnBp3O75bTlStqk5d3YcbNmyYw9I8fthpWPOCR9e8fSFr5ta+sDLjUmbqdj5eOjMZq3zZZuTWBwTZV1Cz1GZoH9eNf5j772450fv55JNPetxnC1oHXv2trg0zB3ogEiZ\/5iAmSG3r\/Jlx2P2bW79pay2obu2c6RjtCzU\/f9L3sJW+Te3qOnPrY7QGctOwQebGvQrILhK3fZknq5KpFoKYCVLQbNheHbvbFZ9tFqaBK0DmVYruKIPG4bavMNOwQRi7CS7I+bqZpVdhup1PvqL1OldbDHZ9JFkHYac43cSk9zVmzBi68cYbXRcp2bXpxtY+\/yg5MQ1MXUWb5m1qhXtv22vmx8yVNmd71sTLoG2d5tOLvnDIty9zsU6QfUU1S69ZLzOOIJ26W7277VuPbuyLB21A5ijXrtd8+VOzXapjVx2+24VqkG1Vnxyktr3yp7ZVIz4129bU1HTeiD1Mv+l2YRUkRp0zO1a3bYNcMLHM0i4YewrFvlLVqyLdOlGdWHO1mN25BDEXe7pMXWnccccdTrHYi13sRKnpJb84vPbllkA\/QZlTc\/YVkVdi7X26na8aOdvs8hVlEK7mcb0ukuyOwG7n999eF0NB6iCsWfpNrXl1Avbx3EzXzyy5OTFzbddL1JGlbbJmR6pnBWz9eNWNfd5ejL3uv+WbFg+yr3379rFWw7rt047Nr07ctB6kJtR2nDqw9+l1Aew2q2PnL8i2ahV7kPPw60O8LhLMaVDOfV23kaW54tktH17+5HZ+bLPUxnHdddc5V9f63pCeJ1adsjkfrOeJ1b\/bKzjNk1P7Ma\/S7A7fPlHdMXglxL53pqYGTYGruWi3kattcNosvUxOxZFvX+ZVFOfRETuJZkeYbxrWnEe3OzQvszTvz5jnybl34HVFah\/TzVSTqoMwV6hBtglq4Pk6FHs6THdatinYdWfnRLffuXPneUvlo5jlnDlznHt3agpP\/+LsbFXO1c++n+xmSPp+pZ7+so0vyL7iMEuTp+7\/uI9A5ZvNMvfFqQM3szTXi9g1ZPeFZg7Mvtutz9O3WILUtt90qn3P3j6efW\/ab6AR1CyD+FOsZun1bJBXp+l1T00b5DPPPNNjUY3fVYkG59bOq8MzO27OyNJvqiXfCCbM1addFPbcu80yyPnmG1nG8ShJ2JGlOlczh4Wog3yiC2KWcYws\/cwyaE5M4Qe5b+\/X4ah\/9zq\/Qo8slcG5zerYmi\/UyNLUuXrG2a9fiGNkGaQO3MzSb0Tl1Wd5jezd+iT7GPZIMahZci84vGo4iFkG9aeCmKXXVbcWtbkYQCdMr6gzH9D163w1YM49APv+GeeepZ1Qt2kl8z6C31RKkE5Lt7E78NbW1h6jdK\/pDPt8ve6PuQkrqGi8YtSjaL97lianuOsgiPEF7dDMGQc16lIr5Pzupwe5+o6SE5OtXv1qriyOMrJ0W6Xsds\/S7syD3vMKcp\/Rq7NV25qLRoLsK+o9S10nOqdqTUMQI7PrK19\/ZfcxXhr0M0e\/ix01va7vWXrlz2sKXv99kNoOcs\/ZvjjWvMIMMoKYZVB\/KohZ2p2jeaWqpgjtlXPaRN2G3G4Gawsj38jSXBloTmvoKU23WPX0pzntGSQON8PS23GW7bsl0+6AvO57+Z2vPcWsDM1txZ1Xh+dn8Dp2t9Ww9uMSbh2DmjpKog5U3JxHd4KsQrRrQteXvRrW7+o7bE68jmcuCAmzWtPUq\/r\/+o03QS68zI7ObSWmubrYTWe2Cdr1pGch1GjTnI4Oui\/OG3y8jMqcOgzzeJWbgfgNDExubtp0i9VmaV84KpZuazZ0PZr5c7voDGKWbvuy+0m3PtjL0Pz6nyBmGdSfCmKWtjmq\/1YdoLoiUcXqdTVuPr5hQrHvXbo9Z+k2HeI256+K235eyp43189ZqhjMuXy\/OFR7e18qVvWMj3rWktNZu92vtZ9N0h2ENhf9OI7f82Hmvu1HYextw7yGy75XrJ+zVKtI1cIAr47BrYij1kHYkaU52tVMgjxnGWSJuteqSv0scdCc6Py7zdSY9y\/tOtGrvvN1PDoX6p6+fu5PtTdr0G\/xlH3vy61+1cWz+Zyl24WSm+5uueUWuu2223LP5+pOMt++uCNLs47dYrdnkfw6cvPf7Xus+S5qvPoxU5v5RqDmvUuzPrn5C1vb6rz9asHuNzlrJWyu5vPxXlzsPsrNnyKbJacg0tpWF5Hu3NN6HlLi1kKI4x2YUs6p1OMIO31b6lzMi6gwU7CaT9D1GFngWWrnWLKvu3O71+F3U7rUkhvX+XiN4twYx3VM7CcZAjBLb66KjdtKYW4mvFbwcveD9rIIlKxZuk1xcJcry0pV\/NH4PcKjj6imqvTD+ubjRJz7tfFH\/8Ue7akXr2N5PduaZGyF2Lffowg6BlX\/99xzD915553sF\/UX4jyKdQyzfsLcq7TjxoxLsTKZ7HFL1iyTxYa9gwAIgAAIZIkAzDJL2ca5ggAIgAAIhCIAswyFDRuBAAiAAAhkiQDMMkvZxrmCAAiAAAiEIgCzDIUNG4EACIAACGSJAMwyS9nGuYIACIAACIQiALMMhQ0bgQAIgAAIZIkAzDJL2ca5ggAIgAAIhCIAswyFDRuBAAiAAAhkiQDMMkvZxrmCAAiAAAiEIgCzDIUNG4EACIAACGSJAMyyBLKt3vF66NAh5zNZUX5B9hP2e3RR4sK2IFBMAkF0ESS+fPtx+5yg+RnBIPtHm2QJwCyT5Zv43sN+zNkOLMh+zG\/XxfHC6cTh4AAgEJFAEF0EOYTffswv+Bw+fJiqq6vxsvsgYAvYBmZZQNjmVz7sj6CaRmR+HcX8PJb+AK7+dxW6+oiv\/hKI\/qqGeRxzX+bX6SdNmuR8Tf29996jJUuW0F133XXefkw09kdeYZYFLBwcKhCBNOvLPEG3j4gHAoBGiRKAWSaK94udm98RVAamjEr9GhoaaP369bRs2TJas2YNDRs2rMe\/qTaq7c6dO51\/Vz911Xn77bc70672Fav539oQ9XH69etH6gp2w4YNdMMNN5AyPG16fle+yiyHDx\/uTPeqL7TDLAtUODhMIAJp15fbhan6NF7UWyuB4KFRIAIwy0CYojdSZvP444\/T6tWraeDAgbkdun1Y2bwvqI1VjQDVtno0OXr0aFL3NGyTsz9wbR\/X\/Haf+T1KP7PUAesRJswyek1gD\/ERKBV9ae3bM0\/xkcKewhKAWYYlx9zOvCdhmqXblIubWerRoTJXNfXqZZbqOCtXruwRnS083cY0PJglM6FoLopAKejLvJDFqFJUeTnBwCwLlBN7xKcPG3RkGdQsvY6jj+d1PxNmWaBCwGESIZB2fZmrYfXag0RAYaehCcAsQ6PjbWibkXklvGnTJt97lkHNMt89Sz0qvfrqq+mrX\/0q3XrrraSFCbPk5ROtZRFIu770bI95a0QWYUQDsyxgDQRdrWcKxh552tOw+t\/Voh29nVodW1VV5ZyZuRrWvt+oF\/uYC4vM\/agFQfYP9ywLWDA4FItAWvVlP2OpTxojTFb6E28Ms0wcMQ4AAiAAAiCQdgIwy7RnEPGDAAiAAAgkTgBmmThiHAAEQAAEQCAoAfPWktpGr9o3\/97tZSuqbZKriGGWQTOIdiAAAiAAAokTUOsi1E+9kEHdz124cCEtWLCA1EJI\/ffq\/nRzczPV1NTk\/l2tsVi0aBFNmzaNRo4cGXucmTPL1tbW2CFih4UjoN5wFPQXJdec4wSNJwvtojDPAh\/p58it+yj5DnIsNZpsampyTFGNMMePH+8YoTLRVatW0YQJExzTnDlzpoNWv2lMPYce9y9TZqkSe+PYm+l09ztxc8T+CkSgb9m1tKn5Uee1gPl+KtezZ3+ZduzoFSqyP\/uzTlq69EXf44TaeYluBH2lP7FB9aXONGmN6WnXKVOmOC9hWb58eQ+zVCPOqVOn0rZt23qYpR59xp2NTJmlXlp+Wa+\/poqyy2Jh2dF9lDq7j1Lf8mtj2V+UnZR6LKe73qFjXZvphRde8DUxneu6ug4aOqSbhXXHy2X0wx9WBDoOa8cl3hj6KlyCk9A6R1\/qTKNo7Oc\/L6dnf97LU2Nq5Dhr1iyaP39+bkrVNkuMLBOsN53cKyvmxGaWCYaLXVsE1IzAJx2rApmYzvWmjeHM8uZbYJbcAoS+uMRktefoyzTLMBr7+S\/Kaf497mapXjBRV1dHjY2NPd6jbd7LxD3LhGsHYk4YcMK754hZ5\/pXv+yioUN4gb38ShndfGtZIFPm7bm0W0Nf6c4vR1+mWcatMfv91nrlq1rAo77ApF6cgtWwCdcaxJww4IR3zxGzzvXG9WU09CpeYC\/vJKr+VjfMkoctNy2HmRsmOCHNOfoyzTIrGsvkPUuIWYg6mWFwxJwzy2d78c3y1W6q\/nYXzJKZH1yMMoEJa87RVw+zzIjGYJbCChbheBPgiDlnlk9X0NCrylhYX36tm6pndsAsWdS+WPCBi1EmOCHNOfrqYZYZ0RjMUkihIgx\/Ahwx58zyyd409EquWXZR9Z0wS\/+M9GyBkSWXmKz2HH31MMuMaAxmKateEU0eAhwx5xb4\/LQv3yxbuujm757ByJJZjTBLJjBhzTn6Ms3yVxnRGMxSWMEinHinYTf95AIaegVvZLlDmeVdp2GWzGKEWTKBCWse1iyzojGYpbCCRTjxmuXmRy\/km+UbnTR17imYJbMYYZZMYMKahzXLrGgMZimsYBFOvGa55Uf9aejl5Sys29\/spG\/WnoRZsqhhgQ8Tl7jmYc0yKxqDWYorWQTkRYAjZj3KeX7lxXyz\/G0HfePeEzBLZiliZMkEJqw5R1\/mPcusaAxmKaxgEU68I8vnVwygYdyR5a4O+vqCz2GWzGKEWTKBCWse2iwzojGYpbCCRTjxmmXz8ktpWCVzGnb3WapaeBxmySxGmCUTmLDmYc0yKxqDWQorWIQTs1k+PIiGVfI+07V9zxmqWvQpzJJZjDBLJjBhzUObZUY0BrMUVrAIJ2azbKykYYOZZrn3DFXVH4FZMosRZskEJqx5aLPMiMZSY5bq+2bTp0+nlpYWmjhxIjU0NJB6C7350x8LVW+lVz\/zzfTmDWm8jkuYSgOGwxGz7ribl1wRwixPU9WDh\/Oapf1hWrP2ivFFhIAIPZtBX1EJpn97jr7M\/jQpjUkjmhqzVJ9tGTFiBE2aNMn5TIv+erYJ1O2Doea\/48pXWvnx4uGIOWeWDVfSsMEVrANtf+s0VS35JK9Z6s8IrVu3zvmKe7G\/tcc6QZfG0FdUgunfnqOvHmaZkMakEU2FWeqr3rlz5+Y6poMHD5L6b\/Pn9dFQ3QZmKa38ePFwxKxzvbnuKho6iGeWO\/adpqmNH3uapf7wrIp+7NixTk0W+yvuPJI9W0NfUeiVzrYcfZlmmYTGJFJNjVnOmjWL5s+fTyNHjnSu4rdu3XreVKz6+3nz5uU419fX0+TJk3P\/DbOUWILBY+KIOfci9boraOgg3j3LFRuO04p\/dl8Nqy7I1q5dS7W1tdTU1ORplgsXLqSpU6fStm3baObMmc5JmqPP4GedfEt7Rgb6Sp65xCNw9GWa5cYQGnt532mqbsx\/q0Mao5IySxOufbVsJndA+Tga0Osr0nKBeHwIHOt8jo51bQ608Eab5S\/rBtEQplmu33aK7nv0mOtx7AuyIUOG0Jo1a2jjxo00fvx452JO1d6qVatowoQJ1Nzc3MMshw8f7oxEJf2CmiX0JSlr8cfC0ZfZn4bR2Cv7ztCtjelacZ4as1SLe\/ymYc3y0QsuxowZkxtd6g60sqKG+pZdG3+1YY+JEjjRtZOOdj7NMsv1dZfQVYN4z1nu3NdB32r0f85S3efT07BpvmcZdBoW+kq0vIu+c46+TLNMUmNFh2IEkAqzVPEGWYBgdlhqumzOnDm0ePFi52rfTC5Ww0oqweCxcKaJ9IXR0\/UXss3ytbc6aWZju68pm2aZ9tWw0FfwOizVlhx9mf1pkhqTxDo1ZmkubZ8xY0ZucY9pkPajI7hnKanUosfCEbM2y8cbLqArB\/E+0fX6W130naXZ+kQX9BW9PtO+B46+TLPMisZSY5ZxFCIW+MRBsXj74IhZ5\/onDX3oisE8s2x5q4vuWnLWd2RZPBIyjwx9ycxL0Kg4+jLNMimNmYvp1DP15gWdOr4eNOm6U39nD5CCnnuQdjDLIJTQRgQBjpi1gH60uDddzjTLN\/d2Ue3iDpglM+swSyYwYc05+jLNMgmN6VoyX0Bjm6eKQRmoWnm+YMEC5yU1ixYtomnTpuVuvcWJGGYZJ03sK1ECHDFrsf39Er5Z7trbRfc9CLPkJhNmySUmqz1HX6ZZxq0xdTtty5YtjuHpx7SUEZojSL0KXcWhVqKbj2clteIcZimrXhFNHgIcMWthNS3tS5XMkeXuvV30QMMZjCyZ1QizZAIT1pyjL9Msk9KYPZK0F3AqI1XP0duPZ6nYzOfr48IMs4yLJPaTOAGOmHXH3fjQhTSYaZZ793RSff0pmCUzozBLJjBhzTn6Ms0yjMaeffYM\/fzZ\/OsC3KZdNTI9\/er24g+MLGMoLIg5BohF3AVHzDrX3394AA0azHvOct+eDnq4Dh9\/5qYa+uISk9Weoy\/TLMNobNtvztCP\/9\/JvBektlmaj2rpV07W1NTgnmUSZQQxJ0G1cPvkiFnnesEPLqOBzO9Zvr3nLC37\/mcYWTJTC30xgQlrztGXaZZJaSzfathifNkH07DCChbheBPgiFl33PMeqaTLmGZ5YPcZWvl9fM+SW4swSy4xWe05+jLNMisag1nKqldEk4cAR8y6457VdBVdWsn76si7u0\/TPz7g\/dURJMmdAMwy3ZXB0ZdpllnRGMwy3fWdqeg5YtYd97eXjaBLKnuzOL23u51++r1WTMOyqFFuaT9eJ8kEJ6Q5R1+mWWZFYzBLIYWKMPwJcMSszfLW5SNpANMs3999gp5aeAhm6Z+SHi0wsmQCE9acoy\/TLLOiMZilsIJFOPHes\/zmD\/8zXVzZh4X1g12f0y8Wvg2zZFHDyJKJS1zzsGaZFY3BLMWVLALyIsARsx7lTF7xJ3RRZV8W1A93Hadf3r8XZsmiBrNk4hLXnKMvc2SZFY3BLMWVLAKK0ywnrvhT6l95AQvqx7s+o1\/f\/wbMkkUNZsnEJa55WLPMisZgluJKFgHFaZbjV3yZLrycZ5a\/33WUfrPgVZglsxRxz5IJTFjzsGaZFY3BLIUVLMKJ957l\/1oxlvpd3o+F9fCuI7R9wQ5XszS\/mapf5qxe+Jz2jz+zAHk0hlnGQbF4+whrlnFrrHgE8h8ZZik1M4jrPAIcMeuOe\/SK8XTB5ReyaH666\/f0+oLfuJql2u+hQ4ecFzXrV27NnTuXzJc8F+NVXKwTTKgxzDIhsAXaLUdf5j3LuDVWoNNlHyZRs1SvK6qurqa2tjYyr8JVlOZ3yAYOHMgOPMwGEHMYanK24YhZ5\/r6FX9BfSv7s07i2K6Padf9z\/tOw5pmuXz5cho\/frzzWSFV26tWraIJEyac90WEOF\/yDH2x0orGPgQ4+jLNMkmNSUpaYmapp6XGjBmTuwqvqqrKfckaZimpDNIRC0fM2iz\/eMVfUh+mWbY9+QZ9+NSbnmapa3vnzp20Zs0axyBts1QfpHX7IoIiHcfng6CvdNRsmqLk6Ms0yzAaO77rI9p3\/3O+F6SS+CVmlm5mqP5u+vTpdNNNN9GNN96Ye1s8RpaSSkJuLBwxa7P8Tyv+mvpUXsQ6qaPPv02tP\/x3XyErw2pqaiL15YN169YVdGQJfbFSisYBCHD0ZZplGI2d2PU7euf+f\/PVWICwC9YkMbPUV75Tpkyh0aNH505IG+a4ceNITSMtWLCAYJYFy3eqD8QRszbL4T\/8OvWuvJh13u272uiDhetdhWzemzQNa9OmTc4xzHuZSX4+CPpipRSNAxDg6Ms0y7g1FiDUojRJzCzV2ZgjSXvqSX2bTHVoq1evhlkWJfXpOyhHzNoshy6fShWVA1gne2r3B\/TRwmd9V8OqnaoRpboYLMZqWOiLldZAjcvKejntysrOvfWpmzrP\/dl1JtD2aW7E0ZdplnFrTCrDRM1S2kljgY+0jPDi4YhZ5\/qKZdOpF9MsT+9upcPfeypVU0Q8ksm0LgV9wSxXBa77rGkMZplMv4G9JkAgjFkObvpb6lV5CSuaM7vfp6MPPB6402DtvIQbp9ksy8vPPYvb59I\/dP4sHzbsXKZOnnb+ONu22\/mzo\/1D6u4+N9ostR9HX+bIMisag1mWWsWX8PlwxKw77ksfmUnllZeyqJzdfYiOf\/8nMEsWtXS\/7g5mScTRl2mWWdEYzJLZIaB58QhwxKzNcsDD32GbZceeg\/T5okdhlsxUp3lk2efikc7Zfvw350aW\/\/UrC5w\/XzzxZefP\/\/gPf+78eeaNLdR59jMmmXQ05+jLNMusaKzkzFKtVpw3b55TnXrxhS7VNIs5HXJLNkqOmHWu+z88i8oHX8YKrHPPu3SybjXM0oVaqeoLZhl+ZJkVjRXELHXHZWtv1KhRsa6GVY+i1NXVUWNjI+3bt48ee+wxamhooH79zt2PgFmyPENc4zBm2e+hOVTGNst36HR98IUOxQYFfYXPgF7Qc8Hg652dbLrnsPPnqXETnD97v3jduZHmi792\/hz81A46294W\/oCCt+Toy+xPs6Axdb6Jm6Ve3q7en2k+b5lEzair3q1btzoGqZbyz5o1i+bPn++8YQVmmQTxwu6TI2ZtIL0aa6nx7MjtAAAWYUlEQVRsMO91it17D1Bn\/YpUjCyhr2g1CLP8gh9HX2Z\/mpTG1OBn7dq1VFtb6wx4ivF4llldBTFL9eqvQrx8QJnlwYMHSRmzWyeCkWW0jqXYW3PErHPdvfQ+IqZZ0t63qaxheWrMEvqKXpl9LrrW2cknfzXK+fPa\/7PI+XNP+x87fw7\/0VedP8+8uoU6zxyNfkCBe+DoyzTLJDSm9Ttx4sTc7GCxP1aQuFkqqKaJJVkjMMsk6RZ\/3xwxa7GdWbKQugcPYgVfvnc\/9X7wkVSYJfTFSq1nY5hl+HuWcWtMjSC3bNnizAiaI8tifKygoCNLLWa96MY8eNz3LINOww4oH0cDen0lHpVhLwUjcKzzOTrWtTmQiWmz\/OzBOupimmXFW\/vo4sWNgY5TsJPPcyBz0Q30FS4jZeXn3tjTZ8C5e5S9hg4\/t6OzHc4fHR\/sPfefJ1tL9jlLjr7MkWVSGrOnYQv9sQK7khIfWarpUPveYbhy9t8q6AKfyooa6lt2btoFv\/QQONG1k452Ph3IxLRZftLQQJ2DB7NOss9bb9HAJUsCHYe14wQaQ1\/xQrXf4KP33t197nV3pfpCAnVuHH2ZZhlGYxetX0\/qfy+88AIN0y+AsFLpZ5aF+AyeGVLiZml+maEQL0zXV9n29zPN5F5ZMYcqyniPE8QrSewtDIEw07Af1i+ljkE8s+z71l66vLEhFWYJfYWpJO9tsmyWHH2Z\/WkYjfXf+u808NEfscwyE\/cszY\/kxlvavL1hgQ+Pl7TWHDHrXLfWPcw2ywv27aErG+tSYZa602pubnYWthXzB30Vk370Y3P0ZZplUhrL5GpY9Q3LlpaW87IZ9z1Lv3KBmP0Iyf53jph1rt\/\/\/jLqGFTJOrEL9u2mqx7+XirMUq\/6hr5YKUZjFwIcfZlmWeoa06gSn4aVVJUwS0nZ4MfCEbPO9QcPrAhhlrvoih\/cnwqz5FNMbgvoKzm2hdgzR1+mWWZFY4mZpf4w7h133EGzZ8\/GyLIQ1V7ix+CIWXfcv1v4D9Q56HIWmb77f0uDH7lPtFlCX6yUonEAAhx9mWZZqhqzkSVmlgFyU\/AmuPItOPJYD8gRs8714ftXUedAnln23v9bumxZrfiPP8cKN4adQV8xQCziLjj6Ms0ybo0VEUHeQ8MspWYGcZ1HgCNm3XEfv+8fqWvgFSyaFW+\/Sf2X3+1qluaKPD26U2+n2rRpk3OMyZMnO+8gVgtuampqSL9dR72ua9GiRTRt2rTc6xdZQaWgMcwyBUnKEyJHX6ZZxq0xqRQTM8t8Cw80DCzwkVoWMuPiiFl33CfvWUvdTLPs9fYbdMGKOb7TsOZjG+oLN+PHj3eMUNV+0s+AQV8yazTNUXH0ZZplkhqTxDMxszRP8sEHH6QRI0Y4V936V4zHSXDlK6n0+LFwxKxzfXr+P7HNsuJXP6aKX+X\/+LN+qfOUKVOcDwQU8+0i0Be\/lrDF+QQ4+jLNMozGyg+0UJ+\/n+17QSopT4mbpTlVZb6UwOvvk4QDs0ySbvL75ohZ5\/rsvJ9S92VXsoIrf2UjVTy52FPIbm\/NKdZ7K6EvVmrROIFp2DAaKzvQQr1XfhdmaedDXfmuXLmS6uvrndGl\/u8ZM2YU9EFqmGW6+4owZtkx53EiplmWvfM69Vr1HVchm69UNC\/+ivl2Eegr3XUtJXqOvsyRZdwak8LDjiPxkaU57VpVVeX8p9ur6AoBCGZZCMrJHYMjZp3rrtlP8M3y3dep7Ed3upqlNiZ9lvq+u1rAc\/fdd9OGDRvIvBev41Dt9cViEoTM40BfSRAu\/X1y9GWaZdwak0q6YGYpAQDMUkIWwsfAEXPOPL77JNGlvGlYevc1ojXuZhk++tLfEvpKd445+jLNkjKiMZhluus7U9FzxKw77vLvPMU2y+6Dr1H3o3ek6n6KhEKAWUrIQvgYOPoyzTIrGoNZhq+tTG1pf42hq6u94OfPEbPuuHvd\/jTRpVexYu0+9Bp1\/XgmzJJFjZznS9WtFnzVhwlOSHOOvkyzzIrGYJZCClV6GKk1y9ueobJLuGb5KnX+9DaYJbMoYZZMYMKahzbLjGgMZimsYKWFo03y3fv\/1Antr0bWOX++ecsU6jj1UUHD5YhZd9wVf\/cs3yzfe5U61n0bZsnMLsySCUxYc46+zJFlVjQGsxRWsNLCSbtZ9qlZzzbLrvd30tknvgWzZBYjzJIJTFjzsGaZFY3BLIUVrLRwysr7OCFd9JNdzp\/PjnrE+XPC6F\/Smc\/fKWi4HDHrjvuCWzdQ2YAhrDiVWZ5+6m9hlixquGfJxCWuOUdf5sgyKxqDWYorWVkBpd0sL6z+F7ZZdra+Qqd+VgOzZJYiRpZMYMKahzXLrGgMZimsYKWGU9F3sBNaeUV\/588zJw4VPFSOmHXHfeG0f6Vy5siy84NXqP2Z6TBLZoZhlkxgwppz9GWOLLOiMZilsIKVGk5qzfIbG6n84qEsrJ1tL1P7L6phlixqmIZl4hLXPLRZZkRjMEtxJYuAvAhwxKxHOf2nbAphljvo5D\/fDLNkliJGlkxgwppz9GWOLLOiMZilsIJFON4EOGLOmeXk56j8IubI8sMddPKX02CWzGKEWTKBCWvO0VcPs4xZY\/rzd+o9y+o3ceJEamhocP6\/2\/uXC4URZlko0jhOZAIcMeuO+6K\/+TXbLDt+t4NO\/us3YZbMjMEsmcCENefoyzTLuDXm9dk5ty\/7zJ07t2AUYZYFQ40DRSXAEbPuuAf85fN8s\/xoO32+6RswS2bCYJZMYMKac\/RlmmXcGlOfwauurqa2tjaH0Lp161w\/sL5q1Sq64447SH3xpxA\/mGUhKOMYsRDgiFl33JdOep7K+w9jHf\/sx9vp+OavwyxZ1LDAh4lLXHOOvkyzjFtjSrvNzc3Ot47NUaYyzfHjx9PIkSN7\/L35XdkkocIsk6SLfcdKgCNmbZYD\/6KZeoUwy0+fr4JZMrOHkSUTmLDmHH2ZZhlGYyd2PUIndzX5akzdv1y0aBFNmzaNNm7c2MMsMbL0KCB1hTF9+nRqaWnJ3fC1h9\/2jWHzI7xmcvFVBGEqDRgOR8y646788xeo14W8keWZT16iIy\/kN0s1VbR27Vqqra11poHM2ivGx58DIvRsBn1FJZj+7Tn6MvvTMBprP\/Qz+uyVu1zN0rw3aeps\/fr1DuTJkyc7X7jRo89CkU\/NyFJ9oX7EiBE0adIkZ0XUlClTnHls86cEP2vWLJo\/f74zVLd\/uPItVFklcxyOmHWur7rxBapgmuXp379EH7\/obZZ633qVnjJLt8UHNTU1tHDhQlqwYIFjqPoK2a02kyEWfK\/QV3BWpdqSoy\/TLOPWmHnhOWTIEFqzZo3Tn3tdkBYqH6kwS33Vq+awlUGqjungwYPOnLb5U1chdXV11NjYSG7z2DDLQpVVMsfhiFnneuj\/bmab5anD2+mjbe5mqQS7ZcsWR7zmyHL58uXnTRFNmDDBufqdOXOmA0TV7fDhw8+7yEuGVvC9Ql\/BWZVyS46+TLOMW2NSGafGLM0Ro+p0tm7d6jx7Y07Fqr+fN29ejnV9fb0zZNc\/mKXUMgwWF0fMOtf\/4YZmqujHm4Y9uv8R+vTt\/PdT7GlY2yzViHLq1Km0bdu2Hmapp5GCnXFhWtkzMtBXYbhLOwpHX6ZZhtHYqSPb6cPt6VoXUFJmaRaffbVsJhf3LKXJNFg8HDFrsxz+5efZZnm87Wf08e65eRcf+JmlWnyQppFlkItR6CtYnaa1FUdfZn8aRmPtR7dT2850rTgXaZbmczbqvtCcOXOcq3O\/aVizSPX89pgxY3Kjy9yzd+XjaECvr6S1pjMb97HO5+hY12bfFXQ9hPw\/n6feF\/De4NP+6Xb64LX8z1naZpmme5bQV2YllPfEOfoqhMakZUmkWbpBCrIAwV5FpUx28eLFucU+uRWSFTXUt+xaablAPD4ETnTtpKOdT7PMcsT\/2BLKLFtb8r\/Bp9RWw0JfkB9HX6ZZJqUxaRlJjVmaS9tnzJiRW9xjGqT96AjuWUort2jxcKaJ9IXRNf\/912yzPPnZDmp9M1uvu4O+otVmKWzN0ZdpllnRWGrMMo5ixAKfOCgWbx8cMefM8r89R7378qZhTx7bQa278SJ1bqahLy4xWe05+uphlhnRGMxSVr0imjwEOGLOmeWfbOKb5fEd1LoXn+jiFiPMkktMVnuOvnqYZUY0BrOUVa+IJmazvPa\/bKTefZgjy+Mv0\/v78fFnbjHCLLnEZLUPa5ZZ0RjMUla9Ipq4zfKP\/o169xnC4nry85fp\/QPTAy0kYu24xBvDLNOd4NBmmRGNwSzTXd+Zip4jZt1xX3vdv1Dv3kyzPPEKvX+wBmbJrC6YJROYsOYcfZnTsFnRGMxSWMEiHG8CHDHnzPIP1vPN8uROev+9GTBLZjHCLJnAhDXn6KuHWWZEYzBLYQWLcOI1y2tG\/IJ6976KhfVk+05qbf07mCWLGr5nycQlrnlYs8yKxmCW4koWAXkR4Ig5txp2+DPUu4Jrlq9Sa9ttMEtmKWJkyQQmrDlHX+bI8pqMaAxmKaxgEU7MI8shT1LviitZWE+eeo1aP74TZsmihpElE5e45qHNMiMag1mKK1kEFOfIcsSVj7HNsv3069T6yf+FWTJLESNLJjBhzcOaZVY0BrMUVrAIJ96R5YjL\/4l697qChbX9TAu1Hp4Ns2RRw8iSiUtc89BmmRGNwSzFlSwCinNkOXzQ6hBm+SZ98Ok8mCWzFDGyZAIT1jysWWZFYzBLYQWLcOIdWQ6\/bCVVlF\/Owtp+9rfUduxemCWLGkaWTFzimoc2y4xoDGYprmQRUJwjy6svWU4V5ZUsqO0du+nD4wthlixqMEsmLnHNw5plVjQGsxRXsggoVrO8+OEQZrmHPjyxCGbJLEVMwzKBCWse2iwzojGYpbCCRTjxTsMOu7CBKsoHs7Ce6nyLfte+xNMs1YeSV65cSUOGDKE1a9bkPi7OOkgJNoZZpjupYc0yKxqDWaa7vjMVPUfMuuMe2u8BqigbxOJ0qnMffXT6B65mqfbb3NzsfHx8\/\/79tHbtWqqtraV+\/fqxjlGKjWGW6c4qR1\/qTLOmsUyaZWVFDfWiy2Kr7JNdO6lv+bWx7jNscKUcSycdpU86VgWaHtVCvrzP7VRRNpCFs6P7CH18ZpnrcZ544gkaPnw4jR49mtrb26mpqYlqampo4EDeMVgBpaSxZg59FSZhcWudoy\/TLLOisUyZZWtrK9049mZSV1D4pZNA37Jrac+B53yDP5frajrdfcC3rVuDvmV\/QHsObDrvn2yzXLRoEU2bNg1TsUQEfYUqNVEbBdWXCjprGsuUWeoEqyTjl04Cw4YNI\/W\/ID+V57C59joORpb5yUdhHiSnaJMsAY6+ovanadNY5swy2VLD3kudAO5ZlnqGcX7FJiBVYzDLYlcGjp86AlgNm7qUIeCUEZCoMZhlyooI4YIACIAACBSeAMwyBHO96k9tWl9fT5MnTz5vL+qxgurqampra3P+bcaMGc7jBkn+1P20efPmOYdYt26ds2Iz6Z9UFkmfN\/afHAGpNQV9JZfzNOwZZsnM0pEjR2jWrFk0f\/58Z8u6ujpqbGw879EBJfjHHnuMGhoaCvIMnjJnHcu+ffsKcmypLJgpRXNBBKTWFPQlqEiKFArMkglemaCaT1+9erVjgnfffTdNmTLlvFGcugo9ePBg4qNJHb463tatWx1zVs\/\/aUMfOXIk8wyDN5fKIvgZoKU0AlJrCvqSVimFjwdmyWRujhjVpsosx4wZ02MqVpmV+vsNGzY4ey\/Ea9FMc1ZX59OnT3eMOsmpWKksmClFc0EEpNYU9CWoSIoUCsySCT6ImO1dmlfLSb3pRaqYi8GCmVI0F0QA+voiGVJZCCqXgoYCs\/TBbS42UIt0xo4dG2ga1tytut8xZ84cWrx4cWJvepE8TVRoFgVVEA4WiQD05Y0v6JQ09BWpBANvDLMMjOpcwyALENQ0rPkaNNPIknrhttQFCMVgwUwpmgsiAH19kQypLASVS0FDgVmGwG1eDZuPaKiFP2rkqe4Tmo+OFOKepToNvbS9UMdTx\/RioWJRP\/VYTTFYhEgrNhFCAPrqORVbVVXl\/IXZ10BfhS9WmGXhmeOIIAACIAACKSMAs0xZwhAuCIAACIBA4QnALAvPHEcEARAAARBIGQGYZcoShnBBAARAAAQKTwBmWXjmOCIIgAAIgEDKCMAsU5Iw8\/GTlpaW2N79aj5yktQLE1KCGGFmmAD0leHkBzx1mGVAUMVulsSzmvqRjsrKSuddtzDLYmcZxy8WAeirWOTTc1yYpeBc6efN1HOTN9xwAx0\/ftx5Ubo5smxqaqL+\/fvT5s2bnb9XbxkaMWKE86mufM9bqn3Pnj3b+XrKU0895frlFMFoEBoIRCYAfUVGmKkdwCyFpluP+pYuXUqjRo1yXsyufm5mqUSvRoaHDx92vqE5ceJE5yXq6iUJ6pfvO5qYhhVaAAgrUQLQV6J4S3LnMEuhabW\/h2n+tz2y1IZovh5LfZoryGfCYJZCCwBhJUoA+koUb0nuHGYpNK32PZR8ZqmmXdVr5WCWQpOJsMQRgL7EpUR8QDBLoSniXPnCLIUmEWGJJQB9iU2N2MBglkJTY37A2e+eJcxSaBIRllgC0JfY1IgNDGYpNjU9v+hx77330oEDB6i2tva81bAwS8FJRGhiCZhfN4G+xKZJTGAwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCcAspWYGcYEACIAACIghALMUkwoEAgIgAAIgIJUAzFJqZhAXCIAACICAGAIwSzGpQCAgAAIgAAJSCfx\/TAoqZMJTykQAAAAASUVORK5CYII=","height":277,"width":459}}
%---
