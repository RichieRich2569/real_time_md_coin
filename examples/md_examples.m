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
fprintf('Paths added. RealTimeCOIN on path: %d\n', exist('RealTimeCOIN', 'class') == 8);
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
fprintf('Ensemble (MD): runs = %d, motor trace = %s\n', ens.runs, mat2str(size(tr.motor_output)));
coinviz.trajectory2D(tr.motor_output, [tgtA, tgtB], ...
    'FigName', 'Ensemble MD: run-averaged trajectory', ...
    'Title', 'Run-averaged motor\_output (30 runs, parfor)', 'ObservedXY', obs);

% Live stepping to read context-aligned ensemble summaries at the final trial.
ensLive = RealTimeCOINEnsemble('runs', 30, 'seed', 3, 'state_dim', N, ...
    'max_contexts', 4, 'num_particles', 100);
for t = 1:T, ensLive.observe_q(cues(t)); ensLive.observe_y(obs(:, t)); end
resp = ensLive.responsibilities_vector();
[muE, covE] = ensLive.state_moments();
fprintf('Run-averaged responsibilities: [%s] (sum %.3f)\n', num2str(resp, '%.2f '), sum(resp));
fprintf('Pooled state mean = [% .3f % .3f]\n', muE(1), muE(2));
fprintf('Pooled state covariance:\n'); disp(covE);

% Per-context 2-D state density (reference frame), averaged across runs.
gx = linspace(-0.8, 0.8, 61);
[GX, GY] = meshgrid(gx, gx);
gridPts = [GX(:)'; GY(:)'];
dmap = ensLive.state_given_context_probability(gridPts);   % Map: ctx -> 1 x numel(GX)
ctxKeys = cell2mat(dmap.keys);
figure('Name', 'Ensemble MD: per-context state density');
tiledlayout(1, max(numel(ctxKeys), 1), 'TileSpacing', 'compact');
for i = 1:numel(ctxKeys)
    nexttile;
    imagesc(gx, gx, reshape(dmap(ctxKeys(i)), size(GX)));
    set(gca, 'YDir', 'normal'); axis square; colorbar;
    title(sprintf('context %d', ctxKeys(i)));
    xlabel('dim 1'); ylabel('dim 2');
end
sgtitle('Run-averaged state\_given\_context\_probability (reference frame)');

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
