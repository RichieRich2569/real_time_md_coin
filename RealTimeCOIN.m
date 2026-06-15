classdef RealTimeCOIN < handle
    %REALTIMECOIN Sequential scalar COIN particle filter.
    %
    %   The public API is intentionally small and state-machine like:
    %   observe_q(q) records the cue for the next trial, observe_y(y)
    %   processes the trial feedback, and query methods expose the current
    %   posterior/predictive summaries. Internally, particles are stored in
    %   vectorized arrays following the original COIN.m implementation.

    properties
        num_particles (1,1) double {mustBeInteger,mustBePositive} = 100;
        max_contexts (1,1) double {mustBeInteger,mustBePositive} = 10;

        gamma_context (1,1) double {mustBeNonnegative} = 0.1;
        alpha_context (1,1) double {mustBePositive} = 8.955;
        rho_context (1,1) double {mustBeNonnegative,mustBeLessThan(rho_context,1)} = 0.2501;
        gamma_cue (1,1) double {mustBeNonnegative} = 0.1;
        alpha_cue (1,1) double {mustBePositive} = 25;

        prior_mean_retention (1,1) double = 0.9425;
        prior_mean_drift (1,1) double = 0.0;
        prior_mean_bias (1,1) double = 0.0;
        prior_precision_retention (1,1) double {mustBeNonnegative} = (837.1).^2;
        prior_precision_drift (1,1) double {mustBeNonnegative} = (1.2227e3).^2;
        prior_precision_bias (1,1) double {mustBeNonnegative} = (70).^2;

        sigma_process_noise (1,1) double {mustBeNonnegative} = 0.0089;
        sigma_sensory_noise (1,1) double {mustBeNonnegative} = 0.03;
        sigma_motor_noise (1,1) double {mustBeNonnegative} = 0.0;

        infer_bias (1,1) logical = false;
    end

    properties (Access = private)
        D;
        pending_q = [];
        trial = 0;
        cue_values = [];
        alignment_cache = [];
        state_version = 0;
    end

    properties (Dependent)
        Trial;
    end

    methods
        function obj = RealTimeCOIN(varargin)
            if mod(nargin, 2) ~= 0
                error('RealTimeCOIN:NameValuePairs', 'Arguments must be name/value pairs.');
            end
            for k = 1:2:nargin
                obj.(varargin{k}) = varargin{k+1};
            end
            obj.resetParticles();
        end

        function observe_q(obj, q)
            if isempty(q) || (isnumeric(q) && isscalar(q) && isnan(q))
                obj.pending_q = [];
                return;
            end
            obj.pending_q = q;
        end

        function observe_y(obj, y)
            if isempty(y) || (isnumeric(y) && isscalar(y) && isnan(y))
                y_val = [];
            else
                y_val = y;
            end

            q_val = obj.consumePendingCue();
            obj.predictContext(q_val);
            obj.predictStates();
            obj.predictStateFeedback();
            obj.resampleParticles(y_val, q_val);
            obj.sampleContext(q_val);
            obj.updateBeliefAboutStates(y_val);
            obj.sampleStates(y_val);
            obj.updateSufficientStatistics(y_val, q_val);
            obj.sampleParameters();

            obj.trial = obj.trial + 1;
            obj.invalidateContextAlignment();
        end

        function probs = context_responsibilities(obj)
            weights = obj.contextProbabilityVector("responsibilities");
            probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
            for c = 1:numel(weights)
                if weights(c) > 0
                    probs(c) = weights(c);
                end
            end
        end

        function probs = context_predicted_probabilities(obj)
            weights = obj.contextProbabilityVector("predicted");
            probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
            for c = 1:numel(weights)
                if weights(c) > 0
                    probs(c) = weights(c);
                end
            end
        end

        function densities = state_probability(obj, values)
            values = values(:)';
            densities = zeros(size(values));
            W = obj.D.responsibilities;
            M = obj.D.state_filtered_mean;
            V = obj.D.state_filtered_var;
            for p = 1:obj.num_particles
                for c = 1:(obj.max_contexts+1)
                    if W(c,p) > 0
                        densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
                    end
                end
            end
            densities = densities ./ obj.num_particles;
        end

        function densities = state_feedback_probability(obj, values)
            values = values(:)';
            densities = zeros(size(values));
            W = obj.D.predicted_probabilities;
            M = obj.D.state_feedback_mean;
            V = obj.D.state_feedback_var;
            for p = 1:obj.num_particles
                for c = 1:(obj.max_contexts+1)
                    if W(c,p) > 0
                        densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
                    end
                end
            end
            densities = densities ./ obj.num_particles;
        end

        function densities = state_given_context_probability(obj, values)
            values = values(:)';
            densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
            alignment = obj.ensureContextAlignment();
            M = alignment.global_contexts.state_mean;
            V = alignment.global_contexts.state_var;
            active = obj.activeSummaryContexts();
            active = active(active <= alignment.K);
            if isempty(active) && alignment.K > 0
                active = 1;
            end
            for c = active
                d = obj.normal_pdf(values, M(c), V(c));
                densities(c) = d;
            end
        end

        function set_stationary(obj)
            obj.resetParticles();
            obj.pending_q = [];
            obj.trial = 0;
            obj.cue_values = [];
        end

        function saveModel(obj, filename, setStationary)
            if nargin < 3
                setStationary = true;
            end
            if setStationary
                saved = obj.serializableState();
                obj.set_stationary();
                model = obj.serializableState();
                save(filename, 'model');
                obj.restoreSerializableState(saved);
            else
                model = obj.serializableState();
                save(filename, 'model');
            end
        end

        function loadModel(obj, filename)
            S = load(filename);
            if isfield(S, 'model')
                obj.restoreSerializableState(S.model);
            else
                % Backward compatibility with the first implementation.
                if isfield(S, 'particles')
                    error('RealTimeCOIN:LegacySave', ...
                        ['This save file contains legacy cell particles. ', ...
                         'Create a fresh model or resave with the vectorized implementation.']);
                end
            end
        end

        function val = get.Trial(obj)
            val = obj.trial;
        end

        function p = predicted_context_probabilities(obj)
            p = obj.contextProbabilityVector("predicted");
        end

        function p = responsibilities(obj)
            p = obj.contextProbabilityVector("responsibilities");
        end

        function u = motor_output(obj)
            W = obj.D.predicted_probabilities;
            u = sum(W .* obj.D.state_feedback_mean, 'all') ./ obj.num_particles;
        end

        function [mu, v] = state_moments(obj)
            W = obj.D.predicted_probabilities;
            mu = sum(W .* obj.D.state_mean, 'all') ./ obj.num_particles;
            second = sum(W .* (obj.D.state_var + obj.D.state_mean.^2), 'all') ./ obj.num_particles;
            v = max(second - mu.^2, 0);
        end

        function n = sampled_context_count(obj)
            n = obj.contextProbabilityVector("count");
        end

        function S = diagnostics(obj)
            S = struct();
            alignment = obj.ensureContextAlignment();
            S.trial = obj.trial;
            S.C = alignment.K;
            S.context = obj.globalSampledContexts(alignment);
            S.predicted_probabilities = obj.globalContextWeights(obj.D.predicted_probabilities, alignment);
            S.responsibilities = obj.globalContextWeights(obj.D.responsibilities, alignment);
            S.state_mean = obj.globalContextMatrix(obj.D.state_mean, alignment);
            S.state_var = obj.globalContextMatrix(obj.D.state_var, alignment);
            S.state_feedback_mean = obj.globalContextMatrix(obj.D.state_feedback_mean, alignment);
            S.state_feedback_var = obj.globalContextMatrix(obj.D.state_feedback_var, alignment);
            S.retention = obj.globalContextMatrix(obj.D.retention, alignment);
            S.drift = obj.globalContextMatrix(obj.D.drift, alignment);
            S.bias = obj.globalContextMatrix(obj.D.bias, alignment);
            S.global_transition_probabilities = obj.globalContextMatrix(obj.D.global_transition_probabilities, alignment);
            S.local_transition_matrix = obj.globalTransitionTensor(obj.D.local_transition_matrix, alignment);
            S.global_cue_probabilities = obj.D.global_cue_probabilities(:, alignment.modal_particle_indices);
            S.local_cue_matrix = obj.globalCueTensor(obj.D.local_cue_matrix, alignment);
            S.alignment = alignment;
            S.raw = obj.D;
        end

        function p = predictive_state_feedback_cdf(obj, y, q)
            if nargin < 3
                q = obj.pending_q;
            end
            qLabel = obj.peekCueLabel(q);
            [W, M, V] = obj.previewPredictiveFeedback(qLabel);
            p = sum(W .* obj.normal_cdf(y, M, V), 'all') ./ obj.num_particles;
            p = min(max(p, 0), 1);
        end

        function p = predictive_cue_p_value(obj, q, u)
            if nargin < 3
                u = rand;
            end
            qLabel = obj.peekCueLabel(q);
            if isempty(qLabel)
                p = NaN;
                return;
            end
            [pmf, labels] = obj.previewCuePmf();
            if qLabel > numel(pmf)
                pmf(end+1:qLabel) = 0;
                labels = 1:numel(pmf);
            end
            f = pmf(qLabel);
            Fminus = sum(pmf(labels < qLabel));
            p = min(max(Fminus + u .* f, 0), 1);
        end

        function u = predictive_motor_output(obj, q)
            if nargin < 2
                q = obj.pending_q;
            end
            qLabel = obj.peekCueLabel(q);
            [W, M, ~] = obj.previewPredictiveFeedback(qLabel);
            u = sum(W .* M, 'all') ./ obj.num_particles;
        end

        function alignment = context_alignment(obj)
            alignment = obj.ensureContextAlignment();
        end
    end

    methods (Access = private)
        function resetParticles(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            D = struct();
            D.C = ones(1, P);
            D.context = ones(1, P);
            D.previous_context = ones(1, P);
            D.Q = 0;

            D.n_context = zeros(Cmax, Cmax, P);
            D.n_cue = zeros(Cmax, 1, P);
            D.dynamics_ss_1 = zeros(Cmax, P, 2);
            D.dynamics_ss_2 = zeros(Cmax, P, 2, 2);
            D.bias_ss_1 = zeros(Cmax, P);
            D.bias_ss_2 = zeros(Cmax, P);

            D.global_transition_probabilities = zeros(Cmax, P);
            D.global_transition_probabilities(1,:) = 1;
            D.global_cue_probabilities = ones(1, P);

            D.retention = obj.sampleScalarNormal(obj.prior_mean_retention, ...
                obj.precisionToVariance(obj.prior_precision_retention), [Cmax, P], 0, 1);
            D.drift = obj.sampleScalarNormal(obj.prior_mean_drift, ...
                obj.precisionToVariance(obj.prior_precision_drift), [Cmax, P], -Inf, Inf);
            if obj.infer_bias
                D.bias = obj.sampleScalarNormal(obj.prior_mean_bias, ...
                    obj.precisionToVariance(obj.prior_precision_bias), [Cmax, P], -Inf, Inf);
            else
                D.bias = zeros(Cmax, P);
            end

            D.state_filtered_mean = obj.stationaryStateMean(D.retention, D.drift);
            D.state_filtered_var = obj.stationaryStateVar(D.retention);
            D.previous_state_filtered_mean = D.state_filtered_mean;
            D.previous_state_filtered_var = D.state_filtered_var;
            D.state_mean = D.state_filtered_mean;
            D.state_var = D.state_filtered_var;
            D.state_feedback_mean = D.state_mean + D.bias;
            D.state_feedback_var = D.state_var + obj.observationVariance();

            D.prior_probabilities = zeros(Cmax, P);
            D.prior_probabilities(1,:) = 1;
            D.predicted_probabilities = D.prior_probabilities;
            D.responsibilities = D.prior_probabilities;
            D.probability_cue = ones(Cmax, P);
            D.probability_state_feedback = ones(Cmax, P);
            D.i_resampled = 1:P;

            obj.D = D; %#ok<*PROP>
            obj.updateLocalTransitionMatrix();
            obj.updateLocalCueMatrix();
            obj.invalidateContextAlignment();
        end

        function state = serializableState(obj)
            state = struct();
            props = properties(obj);
            for i = 1:numel(props)
                if ~strcmp(props{i}, 'Trial')
                    state.properties.(props{i}) = obj.(props{i});
                end
            end
            state.D = obj.D;
            state.pending_q = obj.pending_q;
            state.trial = obj.trial;
            state.cue_values = obj.cue_values;
            state.state_version = obj.state_version;
        end

        function restoreSerializableState(obj, state)
            names = fieldnames(state.properties);
            for i = 1:numel(names)
                obj.(names{i}) = state.properties.(names{i});
            end
            obj.D = state.D;
            obj.pending_q = state.pending_q;
            obj.trial = state.trial;
            obj.cue_values = state.cue_values;
            if isfield(state, 'state_version')
                obj.state_version = state.state_version;
            else
                obj.state_version = obj.trial;
            end
            obj.invalidateContextAlignment();
        end

        function q = consumePendingCue(obj)
            raw = obj.pending_q;
            obj.pending_q = [];
            if isempty(raw)
                q = [];
                return;
            end
            idx = find(arrayfun(@(x) isequal(x, raw), obj.cue_values), 1);
            if isempty(idx)
                obj.cue_values(end+1) = raw;
                q = numel(obj.cue_values);
            else
                q = idx;
            end
            obj.ensureCueColumn(q);
        end

        function q = peekCueLabel(obj, raw)
            if isempty(raw)
                q = [];
                return;
            end
            idx = find(arrayfun(@(x) isequal(x, raw), obj.cue_values), 1);
            if isempty(idx)
                q = numel(obj.cue_values) + 1;
            else
                q = idx;
            end
        end

        function [W, M, V] = previewPredictiveFeedback(obj, q)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            prior = zeros(Cmax, P);
            for p = 1:P
                prior(:,p) = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
            end
            prior = obj.normalizeColumns(prior);

            if isempty(q)
                W = prior;
            else
                qCol = min(q, size(obj.D.local_cue_matrix, 2));
                pcue = squeeze(obj.D.local_cue_matrix(:, qCol, :));
                if P == 1
                    pcue = pcue(:);
                end
                W = obj.normalizeColumns(prior .* pcue);
            end

            Mstate = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
            Vstate = obj.D.retention.^2 .* obj.D.state_filtered_var + obj.sigma_process_noise^2;
            for p = 1:P
                novel = min(obj.D.C(p) + 1, Cmax);
                if obj.D.C(p) < obj.max_contexts
                    Mstate(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
                    Vstate(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
                end
            end
            M = Mstate + obj.D.bias;
            V = Vstate + obj.observationVariance();
        end

        function [pmf, labels] = previewCuePmf(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            Qn = size(obj.D.local_cue_matrix, 2);
            labels = 1:Qn;
            pmf = zeros(1, Qn);
            for p = 1:P
                prior = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
                prior = obj.normalizeColumns(prior);
                cueGivenContext = obj.D.local_cue_matrix(:,:,p);
                if size(cueGivenContext,1) < Cmax
                    cueGivenContext(Cmax,Qn) = 0;
                end
                pmf = pmf + (prior' * cueGivenContext);
            end
            pmf = pmf ./ P;
            if sum(pmf) > 0
                pmf = pmf ./ sum(pmf);
            end
        end

        function ensureCueColumn(obj, q)
            if isempty(q)
                return;
            end
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            if size(obj.D.n_cue, 2) < q
                extra = q - size(obj.D.n_cue, 2);
                obj.D.n_cue(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
            end
            if size(obj.D.global_cue_probabilities, 1) < q
                extra = q - size(obj.D.global_cue_probabilities, 1);
                obj.D.global_cue_probabilities(end+1:end+extra, :) = 0;
            end
            if size(obj.D.local_cue_matrix, 2) < q
                extra = q - size(obj.D.local_cue_matrix, 2);
                obj.D.local_cue_matrix(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
            end
        end

        function instantiateCueIfNeeded(obj, q)
            if isempty(q) || q <= obj.D.Q
                return;
            end
            obj.ensureCueColumn(q + 1);
            b = obj.betaSample(ones(1, obj.num_particles), obj.gamma_cue * ones(1, obj.num_particles));
            mass = obj.D.global_cue_probabilities(q, :);
            obj.D.global_cue_probabilities(q+1, :) = mass .* (1 - b);
            obj.D.global_cue_probabilities(q, :) = mass .* b;
            obj.D.Q = q;
        end

        function predictContext(obj, q)
            obj.updateLocalTransitionMatrix();
            P = obj.num_particles;
            Cmax = obj.max_contexts + 1;
            D = obj.D; %#ok<*PROPLC>
            prior = zeros(Cmax, P);
            for p = 1:P
                prior(:,p) = D.local_transition_matrix(D.context(p), :, p)';
            end
            obj.D.prior_probabilities = obj.normalizeColumns(prior);

            if isempty(q)
                obj.D.probability_cue = ones(Cmax, P);
                obj.D.predicted_probabilities = obj.D.prior_probabilities;
            else
                obj.updateLocalCueMatrix();
                pcue = squeeze(obj.D.local_cue_matrix(:, q, :));
                if P == 1
                    pcue = pcue(:);
                end
                obj.D.probability_cue = pcue;
                obj.D.predicted_probabilities = obj.normalizeColumns(obj.D.prior_probabilities .* pcue);
            end
        end

        function predictStates(obj)
            qv = obj.sigma_process_noise^2;
            obj.D.state_mean = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
            obj.D.state_var = obj.D.retention.^2 .* obj.D.state_filtered_var + qv;

            for p = 1:obj.num_particles
                novel = min(obj.D.C(p) + 1, obj.max_contexts + 1);
                if obj.D.C(p) < obj.max_contexts
                    obj.D.state_mean(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
                    obj.D.state_var(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
                end
            end
            obj.D.state_var = max(obj.D.state_var, 0);
        end

        function predictStateFeedback(obj)
            obj.D.state_feedback_mean = obj.D.state_mean + obj.D.bias;
            obj.D.state_feedback_var = obj.D.state_var + obj.observationVariance();
        end

        function resampleParticles(obj, y, q)
            P = obj.num_particles;
            Cmax = obj.max_contexts + 1;
            if isempty(y)
                py = ones(Cmax, P);
            else
                py = obj.normal_pdf(y, obj.D.state_feedback_mean, obj.D.state_feedback_var);
            end
            obj.D.probability_state_feedback = py;

            log_pc = obj.safeLog(obj.D.prior_probabilities);
            if ~isempty(q)
                log_pc = log_pc + obj.safeLog(obj.D.probability_cue);
            end
            if ~isempty(y)
                log_pc = log_pc + obj.safeLog(py);
            end

            l_w = obj.log_sum_exp(log_pc, 1);
            log_resp = log_pc - l_w;
            resp = exp(log_resp);
            resp(~isfinite(resp)) = 0;

            if isempty(y) && isempty(q)
                idx = 1:P;
            else
                weights = exp(l_w - obj.log_sum_exp(l_w(:), 1));
                idx = obj.systematic_resampling(weights(:)');
            end

            obj.D.i_resampled = idx(:)';
            obj.resampleState(idx);
            obj.D.responsibilities = obj.normalizeColumns(resp(:, idx));
        end

        function resampleState(obj, idx)
            fields2 = {'C','context','prior_probabilities','predicted_probabilities', ...
                'probability_cue','probability_state_feedback','global_transition_probabilities', ...
                'retention','drift','bias','state_mean','state_var','state_feedback_mean', ...
                'state_feedback_var','state_filtered_mean','state_filtered_var', ...
                'bias_ss_1','bias_ss_2'};
            for i = 1:numel(fields2)
                f = fields2{i};
                X = obj.D.(f);
                if isvector(X) && numel(X) == obj.num_particles
                    obj.D.(f) = X(idx);
                elseif size(X,2) == obj.num_particles
                    obj.D.(f) = X(:, idx);
                end
            end
            obj.D.n_context = obj.D.n_context(:,:,idx);
            obj.D.n_cue = obj.D.n_cue(:,:,idx);
            obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1(:,idx,:);
            obj.D.dynamics_ss_2 = obj.D.dynamics_ss_2(:,idx,:,:);
            obj.D.global_cue_probabilities = obj.D.global_cue_probabilities(:,idx);
            obj.D.local_transition_matrix = obj.D.local_transition_matrix(:,:,idx);
            obj.D.local_cue_matrix = obj.D.local_cue_matrix(:,:,idx);
            obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
            obj.D.previous_state_filtered_var = obj.D.state_filtered_var;
        end

        function sampleContext(obj, q)
            P = obj.num_particles;
            oldC = obj.D.C;
            obj.D.previous_context = obj.D.context;
            cumResp = cumsum(obj.D.responsibilities, 1);
            r = rand(1, P);
            newContext = sum(r > cumResp, 1) + 1;
            for p = 1:P
                if newContext(p) > obj.D.C(p)
                    if obj.D.C(p) < obj.max_contexts
                        obj.D.C(p) = obj.D.C(p) + 1;
                        newContext(p) = obj.D.C(p);
                    else
                        newContext(p) = obj.D.C(p);
                    end
                end
            end
            obj.D.context = newContext;

            pNew = find(obj.D.C > oldC & obj.D.C < obj.max_contexts);
            if ~isempty(pNew)
                b = obj.betaSample(ones(1, numel(pNew)), obj.gamma_context * ones(1, numel(pNew)));
                for k = 1:numel(pNew)
                    p = pNew(k);
                    c = obj.D.C(p);
                    mass = obj.D.global_transition_probabilities(c, p);
                    obj.D.global_transition_probabilities(c+1, p) = mass .* (1 - b(k));
                    obj.D.global_transition_probabilities(c, p) = mass .* b(k);
                    obj.D.state_filtered_mean(c,p) = obj.stationaryStateMean(obj.D.retention(c,p), obj.D.drift(c,p));
                    obj.D.state_filtered_var(c,p) = obj.stationaryStateVar(obj.D.retention(c,p));
                end
            end
            obj.instantiateCueIfNeeded(q);
        end

        function updateBeliefAboutStates(obj, y)
            obj.D.state_filtered_mean = obj.D.state_mean;
            obj.D.state_filtered_var = obj.D.state_var;
            if isempty(y)
                return;
            end
            obsVar = obj.observationVariance();
            for p = 1:obj.num_particles
                c = obj.D.context(p);
                predVar = obj.D.state_var(c,p);
                totalVar = predVar + obsVar;
                if totalVar <= 0
                    K = 0;
                else
                    K = predVar ./ totalVar;
                end
                innovation = y - obj.D.state_feedback_mean(c,p);
                obj.D.state_filtered_mean(c,p) = obj.D.state_mean(c,p) + K .* innovation;
                obj.D.state_filtered_var(c,p) = max((1 - K) .* predVar, 0);
            end
        end

        function sampleStates(obj, y)
            qVar = obj.sigma_process_noise^2;
            obsVar = obj.observationVariance();
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;

            g = obj.D.retention .* obj.safeDivide(obj.D.previous_state_filtered_var, obj.D.state_var);
            smoothMean = obj.D.previous_state_filtered_mean + g .* (obj.D.state_filtered_mean - obj.D.state_mean);
            smoothVar = obj.D.previous_state_filtered_var + g.^2 .* (obj.D.state_filtered_var - obj.D.state_var);
            smoothVar = max(smoothVar, 0);
            obj.D.previous_x_dynamics = obj.sampleScalarNormal(smoothMean, smoothVar, [Cmax, P], -Inf, Inf);

            dynMean = obj.D.retention .* obj.D.previous_x_dynamics + obj.D.drift;
            if isempty(y)
                obj.D.x_dynamics = obj.sampleScalarNormal(dynMean, qVar, [Cmax, P], -Inf, Inf);
            else
                active = zeros(Cmax, P);
                idx = sub2ind([Cmax, P], obj.D.context, 1:P);
                active(idx) = 1;
                if qVar == 0
                    postMean = dynMean;
                    postVar = zeros(Cmax, P);
                    if obsVar == 0
                        postMean(idx) = y - obj.D.bias(idx);
                    end
                elseif obsVar == 0
                    postMean = dynMean;
                    postVar = qVar * ones(Cmax, P);
                    postMean(idx) = y - obj.D.bias(idx);
                    postVar(idx) = 0;
                else
                    postVar = 1 ./ (1 ./ qVar + active ./ obsVar);
                    postMean = postVar .* (dynMean ./ qVar + active .* (y - obj.D.bias) ./ obsVar);
                end
                obj.D.x_dynamics = obj.sampleScalarNormal(postMean, postVar, [Cmax, P], -Inf, Inf);
            end

            activeIdx = sub2ind([Cmax, P], obj.D.context, 1:P);
            obj.D.x_bias = obj.D.x_dynamics(activeIdx);
            obj.D.i_observed = activeIdx;
        end

        function updateSufficientStatistics(obj, y, q)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            idx = sub2ind([Cmax, Cmax, P], obj.D.previous_context, obj.D.context, 1:P);
            obj.D.n_context(idx) = obj.D.n_context(idx) + 1;

            if ~isempty(q)
                obj.ensureCueColumn(q);
                idxCue = sub2ind(size(obj.D.n_cue), obj.D.context, q * ones(1, P), 1:P);
                obj.D.n_cue(idxCue) = obj.D.n_cue(idxCue) + 1;
            end

            if obj.trial > 0
                xAug = ones(Cmax, P, 2);
                xAug(:,:,1) = obj.D.previous_x_dynamics;
                observedRows = squeeze(sum(obj.D.n_context, 2)) > 0;
                ss1 = obj.D.x_dynamics .* xAug;
                obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1 + ss1 .* observedRows;
                for a = 1:2
                    for b = 1:2
                        obj.D.dynamics_ss_2(:,:,a,b) = obj.D.dynamics_ss_2(:,:,a,b) + ...
                            xAug(:,:,a) .* xAug(:,:,b) .* observedRows;
                    end
                end
            end

            if obj.infer_bias && ~isempty(y)
                obj.D.bias_ss_1(obj.D.i_observed) = obj.D.bias_ss_1(obj.D.i_observed) + (y - obj.D.x_bias);
                obj.D.bias_ss_2(obj.D.i_observed) = obj.D.bias_ss_2(obj.D.i_observed) + 1;
            end
        end

        function sampleParameters(obj)
            obj.sampleGlobalTransitionProbabilities();
            obj.sampleGlobalCueProbabilities();
            obj.sampleDynamics();
            obj.sampleBias();
            obj.updateLocalTransitionMatrix();
            obj.updateLocalCueMatrix();
        end

        function sampleGlobalTransitionProbabilities(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            kappa = obj.kappa();
            m = zeros(Cmax, Cmax, P);
            for p = 1:P
                base = obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' + kappa .* eye(Cmax);
                m(:,:,p) = obj.sample_num_tables(base, obj.D.n_context(:,:,p));
                if obj.rho_context > 0
                    for j = 1:Cmax
                        if m(j,j,p) > 0
                            betaJ = obj.D.global_transition_probabilities(j,p);
                            prob = obj.rho_context ./ max(obj.rho_context + betaJ .* (1 - obj.rho_context), realmin);
                            m(j,j,p) = m(j,j,p) - obj.binomialSample(m(j,j,p), prob);
                        end
                    end
                end
                if m(1,1,p) == 0
                    m(1,1,p) = 1;
                end
                alpha = squeeze(sum(m(:,:,p), 1))';
                if obj.D.C(p) < obj.max_contexts
                    alpha(obj.D.C(p)+1) = obj.gamma_context;
                    alpha(obj.D.C(p)+2:end) = 0;
                else
                    alpha(obj.D.C(p)+1:end) = 0;
                end
                obj.D.global_transition_probabilities(:,p) = obj.dirichletSample(alpha(:));
            end
        end

        function sampleGlobalCueProbabilities(obj)
            if size(obj.D.global_cue_probabilities, 1) == 0
                return;
            end
            P = obj.num_particles;
            Qn = max(obj.D.Q + 1, size(obj.D.global_cue_probabilities, 1));
            obj.ensureCueColumn(Qn);
            for p = 1:P
                counts = obj.D.n_cue(:,1:Qn,p);
                base = repmat(obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)', size(counts,1), 1);
                m = obj.sample_num_tables(base, counts);
                alpha = sum(m, 1);
                alpha(obj.D.Q + 1) = obj.gamma_cue;
                if obj.D.Q + 2 <= Qn
                    alpha(obj.D.Q + 2:end) = 0;
                end
                obj.D.global_cue_probabilities(1:Qn,p) = obj.dirichletSample(alpha(:));
            end
        end

        function sampleDynamics(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            priorPrec = diag([obj.prior_precision_retention, obj.prior_precision_drift]);
            priorMean = [obj.prior_mean_retention; obj.prior_mean_drift];
            qVar = obj.sigma_process_noise^2;
            for p = 1:P
                for c = 1:Cmax
                    ss2 = squeeze(obj.D.dynamics_ss_2(c,p,:,:));
                    ss1 = squeeze(obj.D.dynamics_ss_1(c,p,:));
                    if qVar == 0
                        covar = obj.safeInverse(priorPrec + ss2 ./ eps);
                        mu = covar * (priorPrec * priorMean + ss1 ./ eps);
                    else
                        covar = obj.safeInverse(priorPrec + ss2 ./ qVar);
                        mu = covar * (priorPrec * priorMean + ss1 ./ qVar);
                    end
                    sample = obj.sampleBivariateTruncated(mu, covar);
                    obj.D.retention(c,p) = sample(1);
                    obj.D.drift(c,p) = sample(2);
                    obj.D.dynamics_mean(:,c,p) = mu;
                    obj.D.dynamics_covar(:,:,c,p) = covar;
                end
            end
        end

        function sampleBias(obj)
            if ~obj.infer_bias
                obj.D.bias = zeros(obj.max_contexts+1, obj.num_particles);
                obj.D.bias_mean = obj.D.bias;
                obj.D.bias_var = zeros(size(obj.D.bias));
                return;
            end
            obsVar = obj.observationVariance();
            if obsVar == 0
                obsVar = eps;
            end
            varB = 1 ./ (obj.prior_precision_bias + obj.D.bias_ss_2 ./ obsVar);
            muB = varB .* (obj.prior_precision_bias .* obj.prior_mean_bias + obj.D.bias_ss_1 ./ obsVar);
            obj.D.bias_mean = muB;
            obj.D.bias_var = varB;
            obj.D.bias = obj.sampleScalarNormal(muB, varB, size(muB), -Inf, Inf);
        end

        function updateLocalTransitionMatrix(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            kappa = obj.kappa();
            T = zeros(Cmax, Cmax, P);
            for p = 1:P
                raw = obj.nContextSlice(p) + obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' + kappa .* eye(Cmax);
                valid = false(1, Cmax);
                valid(1:obj.D.C(p)) = true;
                if obj.D.C(p) < obj.max_contexts
                    valid(obj.D.C(p)+1) = true;
                end
                raw(:, ~valid) = 0;
                raw(~valid, :) = 0;
                rowSums = sum(raw, 2);
                for r = 1:Cmax
                    if rowSums(r) > 0
                        T(r,:,p) = raw(r,:) ./ rowSums(r);
                    end
                end
            end
            obj.D.local_transition_matrix = T;
        end

        function updateLocalCueMatrix(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            Qn = max(1, size(obj.D.global_cue_probabilities, 1));
            L = zeros(Cmax, Qn, P);
            for p = 1:P
                counts = obj.D.n_cue(:,1:min(Qn,size(obj.D.n_cue,2)),p);
                if size(counts,2) < Qn
                    counts(:,end+1:Qn) = 0;
                end
                raw = counts + obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)';
                valid = false(Cmax,1);
                valid(1:obj.D.C(p)) = true;
                if obj.D.C(p) < obj.max_contexts
                    valid(obj.D.C(p)+1) = true;
                end
                raw(~valid,:) = 0;
                rowSums = sum(raw, 2);
                for c = 1:Cmax
                    if rowSums(c) > 0
                        L(c,:,p) = raw(c,:) ./ rowSums(c);
                    end
                end
            end
            obj.D.local_cue_matrix = L;
        end

        function A = nContextSlice(obj, p)
            A = obj.D.n_context(:,:,p);
        end

        function invalidateContextAlignment(obj)
            obj.state_version = obj.state_version + 1;
            obj.alignment_cache = [];
        end

        function alignment = ensureContextAlignment(obj)
            if ~isempty(obj.alignment_cache) && ...
                    isfield(obj.alignment_cache, 'cache_state_version') && ...
                    obj.alignment_cache.cache_state_version == obj.state_version
                alignment = obj.alignment_cache;
                return;
            end
            alignment = obj.computeContextAlignment();
            obj.alignment_cache = alignment;
        end

        function alignment = computeContextAlignment(obj)
            Cmax = obj.max_contexts + 1;
            P = obj.num_particles;
            cards = obj.D.C(:)';
            Km = obj.modalCardinality(cards);
            modalMask = cards == Km;
            modalIdx = find(modalMask);
            if isempty(modalIdx)
                modalIdx = 1:P;
                modalMask = true(1, P);
                Km = min(max(cards), obj.max_contexts);
            end

            nModal = numel(modalIdx);
            weights = ones(1, nModal) ./ nModal;
            assignment = zeros(Cmax, P);
            anchor = modalIdx(1);
            assignment(1:Km, anchor) = 1:Km;
            if Km < obj.max_contexts
                assignment(Km+1, anchor) = Km + 1;
            end

            prototypes = obj.updateGlobalContexts(Km, anchor, 1, assignment);
            oldAssignments = zeros(Cmax, P);
            converged = false;
            includeTransition = false;
            maxIterations = 20;

            for iter = 1:maxIterations
                for idx = 1:nModal
                    p = modalIdx(idx);
                    cost = obj.assignmentCostMatrix(p, Km, prototypes, assignment, includeTransition);
                    perm = obj.minAssignment(cost);
                    assignment(:,p) = 0;
                    assignment(1:Km,p) = perm(:);
                    if Km < obj.max_contexts
                        assignment(Km+1,p) = Km + 1;
                    end
                end

                prototypes = obj.updateGlobalContexts(Km, modalIdx, weights, assignment);
                if iter > 1 && isequal(assignment(1:Km,modalIdx), oldAssignments(1:Km,modalIdx))
                    converged = true;
                    break;
                end
                oldAssignments = assignment;
                includeTransition = true;
            end

            alignment = struct();
            alignment.K = Km;
            alignment.assignment = assignment;
            alignment.modal_particle_mask = modalMask;
            alignment.modal_particle_indices = modalIdx;
            alignment.modal_particle_weights = weights;
            alignment.global_contexts = prototypes;
            alignment.converged = converged;
            alignment.iterations = iter;
            alignment.cache_state_version = obj.state_version;
            alignment.computed_at_trial = obj.trial;
        end

        function cost = assignmentCostMatrix(obj, p, Km, prototypes, assignment, includeTransition)
            cost = zeros(Km, Km);
            for local = 1:Km
                [dynMean, dynCovar] = obj.localDynamicsDistribution(local, p);
                cueProb = obj.localCueProbability(local, p);
                if includeTransition
                    transitionProb = obj.globalTransitionRow(local, p, Km, assignment);
                end
                for globalIdx = 1:Km
                    total = obj.gaussianJeffreys(obj.D.state_filtered_mean(local,p), ...
                        obj.D.state_filtered_var(local,p), ...
                        prototypes.state_mean(globalIdx), prototypes.state_var(globalIdx));
                    total = total + obj.gaussianJeffreysMulti(dynMean, dynCovar, ...
                        prototypes.dynamics_mean(:,globalIdx), prototypes.dynamics_covar(:,:,globalIdx));
                    if obj.infer_bias
                        [biasMean, biasVar] = obj.localBiasDistribution(local, p);
                        total = total + obj.gaussianJeffreys(biasMean, biasVar, ...
                            prototypes.bias_mean(globalIdx), prototypes.bias_var(globalIdx));
                    end
                    total = total + obj.categoricalJeffreys(cueProb, prototypes.cue_prob(globalIdx,:));
                    if includeTransition
                        total = total + obj.categoricalJeffreys(transitionProb, prototypes.transition_prob(globalIdx,:));
                    end
                    cost(local,globalIdx) = total;
                end
            end
        end

        function prototypes = updateGlobalContexts(obj, Km, modalIdx, weights, assignment)
            modalIdx = modalIdx(:)';
            weights = weights(:)';
            weights = weights ./ sum(weights);
            Qn = max(1, size(obj.D.local_cue_matrix, 2));
            prototypes = struct();
            prototypes.state_mean = zeros(1, Km);
            prototypes.state_var = zeros(1, Km);
            prototypes.dynamics_mean = zeros(2, Km);
            prototypes.dynamics_covar = zeros(2, 2, Km);
            prototypes.bias_mean = zeros(1, Km);
            prototypes.bias_var = zeros(1, Km);
            prototypes.cue_prob = zeros(Km, Qn);
            prototypes.transition_prob = zeros(Km, Km + 1);

            for globalIdx = 1:Km
                totalWeight = 0;
                stateMean = 0;
                stateSecond = 0;
                dynMean = zeros(2,1);
                dynSecond = zeros(2,2);
                biasMean = 0;
                biasSecond = 0;
                cueAccum = zeros(1, Qn);
                transitionAccum = zeros(1, Km + 1);

                for idx = 1:numel(modalIdx)
                    p = modalIdx(idx);
                    local = find(assignment(1:Km,p) == globalIdx, 1);
                    if isempty(local)
                        continue;
                    end
                    w = weights(idx);
                    totalWeight = totalWeight + w;

                    m = obj.D.state_filtered_mean(local,p);
                    v = max(obj.D.state_filtered_var(local,p), 0);
                    stateMean = stateMean + w .* m;
                    stateSecond = stateSecond + w .* (v + m.^2);

                    [dm, dc] = obj.localDynamicsDistribution(local, p);
                    dynMean = dynMean + w .* dm;
                    dynSecond = dynSecond + w .* (dc + dm * dm');

                    [bm, bv] = obj.localBiasDistribution(local, p);
                    biasMean = biasMean + w .* bm;
                    biasSecond = biasSecond + w .* (bv + bm.^2);

                    cueAccum = cueAccum + w .* obj.localCueProbability(local, p);
                    transitionAccum = transitionAccum + w .* obj.globalTransitionRow(local, p, Km, assignment);
                end

                if totalWeight > 0
                    stateMean = stateMean ./ totalWeight;
                    stateSecond = stateSecond ./ totalWeight;
                    dynMean = dynMean ./ totalWeight;
                    dynSecond = dynSecond ./ totalWeight;
                    biasMean = biasMean ./ totalWeight;
                    biasSecond = biasSecond ./ totalWeight;
                    cueAccum = cueAccum ./ totalWeight;
                    transitionAccum = transitionAccum ./ totalWeight;
                end

                prototypes.state_mean(globalIdx) = stateMean;
                prototypes.state_var(globalIdx) = max(stateSecond - stateMean.^2, 0);
                prototypes.dynamics_mean(:,globalIdx) = dynMean;
                dynCovar = dynSecond - dynMean * dynMean';
                prototypes.dynamics_covar(:,:,globalIdx) = obj.regularizeCovariance(dynCovar);
                prototypes.bias_mean(globalIdx) = biasMean;
                prototypes.bias_var(globalIdx) = max(biasSecond - biasMean.^2, 0);
                prototypes.cue_prob(globalIdx,:) = obj.normalizeProbability(cueAccum);
                prototypes.transition_prob(globalIdx,:) = obj.normalizeProbability(transitionAccum);
            end
        end

        function Wg = globalContextWeights(obj, W, alignment)
            if nargin < 3
                alignment = obj.ensureContextAlignment();
            end
            Cmax = obj.max_contexts + 1;
            Km = alignment.K;
            modalIdx = alignment.modal_particle_indices;
            Wg = zeros(Cmax, numel(modalIdx));
            for idx = 1:numel(modalIdx)
                p = modalIdx(idx);
                for local = 1:Cmax
                    target = alignment.assignment(local,p);
                    if target > 0 && target <= Cmax
                        Wg(target,idx) = Wg(target,idx) + W(local,p);
                    end
                end
                if Km >= obj.max_contexts
                    Wg(Km+1:end,idx) = 0;
                end
            end
        end

        function Xg = globalContextMatrix(obj, X, alignment)
            if nargin < 3
                alignment = obj.ensureContextAlignment();
            end
            Cmax = obj.max_contexts + 1;
            modalIdx = alignment.modal_particle_indices;
            Xg = zeros(Cmax, numel(modalIdx));
            for idx = 1:numel(modalIdx)
                p = modalIdx(idx);
                for local = 1:Cmax
                    target = alignment.assignment(local,p);
                    if target > 0 && target <= Cmax
                        Xg(target,idx) = X(local,p);
                    end
                end
            end
        end

        function Tg = globalTransitionTensor(obj, T, alignment)
            if nargin < 3
                alignment = obj.ensureContextAlignment();
            end
            Cmax = obj.max_contexts + 1;
            modalIdx = alignment.modal_particle_indices;
            Tg = zeros(Cmax, Cmax, numel(modalIdx));
            for idx = 1:numel(modalIdx)
                p = modalIdx(idx);
                for localFrom = 1:Cmax
                    globalFrom = alignment.assignment(localFrom,p);
                    if globalFrom <= 0 || globalFrom > Cmax
                        continue;
                    end
                    for localTo = 1:Cmax
                        globalTo = alignment.assignment(localTo,p);
                        if globalTo > 0 && globalTo <= Cmax
                            Tg(globalFrom,globalTo,idx) = Tg(globalFrom,globalTo,idx) + T(localFrom,localTo,p);
                        end
                    end
                end
            end
        end

        function Lg = globalCueTensor(obj, L, alignment)
            if nargin < 3
                alignment = obj.ensureContextAlignment();
            end
            Cmax = obj.max_contexts + 1;
            modalIdx = alignment.modal_particle_indices;
            Lg = zeros(Cmax, size(L,2), numel(modalIdx));
            for idx = 1:numel(modalIdx)
                p = modalIdx(idx);
                for local = 1:Cmax
                    target = alignment.assignment(local,p);
                    if target > 0 && target <= Cmax
                        Lg(target,:,idx) = Lg(target,:,idx) + L(local,:,p);
                    end
                end
            end
        end

        function c = globalSampledContexts(obj, alignment)
            if nargin < 2
                alignment = obj.ensureContextAlignment();
            end
            modalIdx = alignment.modal_particle_indices;
            c = NaN(1, numel(modalIdx));
            for idx = 1:numel(modalIdx)
                p = modalIdx(idx);
                local = obj.D.context(p);
                if local <= size(alignment.assignment, 1)
                    target = alignment.assignment(local,p);
                    if target > 0
                        c(idx) = target;
                    end
                end
            end
        end

        function row = globalTransitionRow(obj, local, p, Km, assignment)
            row = zeros(1, Km + 1);
            maxLocal = Km;
            if Km < obj.max_contexts
                maxLocal = Km + 1;
            end
            for dest = 1:maxLocal
                target = assignment(dest,p);
                if target > 0 && target <= Km + 1
                    row(target) = row(target) + obj.D.local_transition_matrix(local,dest,p);
                end
            end
            row = obj.normalizeProbability(row);
        end

        function q = localCueProbability(obj, local, p)
            q = squeeze(obj.D.local_cue_matrix(local,:,p));
            q = obj.normalizeProbability(q(:)');
        end

        function [mu, covar] = localDynamicsDistribution(obj, local, p)
            if isfield(obj.D, 'dynamics_mean') && ...
                    size(obj.D.dynamics_mean, 2) >= local && size(obj.D.dynamics_mean, 3) >= p
                mu = obj.D.dynamics_mean(:,local,p);
            else
                mu = [obj.D.retention(local,p); obj.D.drift(local,p)];
            end

            if isfield(obj.D, 'dynamics_covar') && ...
                    size(obj.D.dynamics_covar, 3) >= local && size(obj.D.dynamics_covar, 4) >= p
                covar = obj.D.dynamics_covar(:,:,local,p);
            else
                covar = diag([obj.precisionToVariance(obj.prior_precision_retention), ...
                    obj.precisionToVariance(obj.prior_precision_drift)]);
            end
            covar = obj.regularizeCovariance(covar);
        end

        function [mu, variance] = localBiasDistribution(obj, local, p)
            if obj.infer_bias && isfield(obj.D, 'bias_mean') && ...
                    size(obj.D.bias_mean, 1) >= local && size(obj.D.bias_mean, 2) >= p
                mu = obj.D.bias_mean(local,p);
                variance = obj.D.bias_var(local,p);
            elseif obj.infer_bias
                mu = obj.D.bias(local,p);
                variance = obj.precisionToVariance(obj.prior_precision_bias);
            else
                mu = 0;
                variance = 0;
            end
            if ~isfinite(variance)
                variance = 1 ./ eps;
            end
            variance = max(variance, 0);
        end

        function weights = contextProbabilityVector(obj, kind)
            alignment = obj.ensureContextAlignment();
            switch kind
                case "predicted"
                    W = obj.globalContextWeights(obj.D.predicted_probabilities, alignment);
                    weights = mean(W, 2)';
                case "responsibilities"
                    W = obj.globalContextWeights(obj.D.responsibilities, alignment);
                    weights = mean(W, 2)';
                case "count"
                    c = obj.globalSampledContexts(alignment);
                    weights = histcounts(c, 0.5:1:(obj.max_contexts+1.5)) ./ max(numel(c), 1);
            end
            weights(~isfinite(weights)) = 0;
            s = sum(weights);
            if s > 0
                weights = weights ./ s;
            end
        end

        function active = activeSummaryContexts(obj)
            weights = obj.contextProbabilityVector("predicted");
            active = find(weights > 0);
            if isempty(active)
                active = 1;
            end
        end

        function Km = modalCardinality(~, cards)
            vals = unique(cards(:)');
            counts = arrayfun(@(v) sum(cards == v), vals);
            best = counts == max(counts);
            Km = min(vals(best));
        end

        function assignment = minAssignment(~, cost)
            n = size(cost, 1);
            nMasks = 2^n;
            dp = Inf(n+1, nMasks);
            parent = zeros(n+1, nMasks);
            dp(1,1) = 0;
            for source = 1:n
                for mask = 0:(nMasks-1)
                    current = dp(source, mask+1);
                    if ~isfinite(current)
                        continue;
                    end
                    for target = 1:n
                        bit = bitshift(1, target-1);
                        if bitand(mask, bit) == 0
                            nextMask = bitor(mask, bit);
                            candidate = current + cost(source,target);
                            if candidate < dp(source+1, nextMask+1)
                                dp(source+1, nextMask+1) = candidate;
                                parent(source+1, nextMask+1) = target;
                            end
                        end
                    end
                end
            end
            assignment = zeros(1,n);
            mask = nMasks - 1;
            for source = n:-1:1
                target = parent(source+1, mask+1);
                if target == 0
                    target = source;
                end
                assignment(source) = target;
                mask = mask - bitshift(1, target-1);
            end
        end

        function d = gaussianJeffreys(~, m1, v1, m2, v2)
            if ~isfinite(v1)
                v1 = 1 ./ eps;
            end
            if ~isfinite(v2)
                v2 = 1 ./ eps;
            end
            v1 = max(v1, eps);
            v2 = max(v2, eps);
            d = 0.5 .* (v1 ./ v2 + v2 ./ v1 + (m1 - m2).^2 .* (1 ./ v1 + 1 ./ v2) - 2);
            if ~isfinite(d)
                d = realmax;
            end
            d = max(d, 0);
        end

        function d = gaussianJeffreysMulti(obj, m1, s1, m2, s2)
            m1 = m1(:);
            m2 = m2(:);
            s1 = obj.regularizeCovariance(s1);
            s2 = obj.regularizeCovariance(s2);
            inv1 = obj.safeInverse(s1);
            inv2 = obj.safeInverse(s2);
            delta = m1 - m2;
            k = numel(m1);
            d = 0.5 .* (trace(inv2 * s1 + inv1 * s2) + delta' * (inv1 + inv2) * delta - 2 .* k);
            if ~isfinite(d)
                d = realmax;
            end
            d = max(d, 0);
        end

        function d = categoricalJeffreys(obj, p, q)
            n = max(numel(p), numel(q));
            p(end+1:n) = 0;
            q(end+1:n) = 0;
            p = obj.normalizeProbability(p);
            q = obj.normalizeProbability(q);
            d = sum((p - q) .* log(p ./ q));
            if ~isfinite(d)
                d = realmax;
            end
            d = max(d, 0);
        end

        function p = normalizeProbability(~, p)
            p = double(p(:)');
            p(~isfinite(p) | p < 0) = 0;
            s = sum(p);
            if s <= 0
                p = ones(size(p)) ./ max(numel(p), 1);
            else
                p = p ./ s;
            end
            p = max(p, realmin);
            p = p ./ sum(p);
        end

        function covar = regularizeCovariance(~, covar)
            covar(~isfinite(covar)) = 0;
            covar = (covar + covar') ./ 2;
            if isempty(covar)
                covar = eps;
                return;
            end
            covar = covar + eps .* eye(size(covar));
            if rcond(covar) < 1e-12
                covar = covar + 1e-9 .* eye(size(covar));
            end
        end

        function k = kappa(obj)
            k = obj.alpha_context * obj.rho_context / max(1 - obj.rho_context, realmin);
        end

        function v = observationVariance(obj)
            v = obj.sigma_sensory_noise^2 + obj.sigma_motor_noise^2;
        end

        function m = stationaryStateMean(~, a, d)
            denom = 1 - a;
            m = zeros(size(a));
            good = abs(denom) > eps;
            m(good) = d(good) ./ denom(good);
        end

        function v = stationaryStateVar(obj, a)
            denom = 1 - a.^2;
            v = zeros(size(a));
            good = denom > eps;
            v(good) = obj.sigma_process_noise^2 ./ denom(good);
        end

        function X = normalizeColumns(~, X)
            sums = sum(X, 1);
            for p = 1:size(X,2)
                if sums(p) > 0 && isfinite(sums(p))
                    X(:,p) = X(:,p) ./ sums(p);
                else
                    X(:,p) = 0;
                    X(1,p) = 1;
                end
            end
        end

        function y = safeLog(~, x)
            y = log(max(x, realmin));
        end

        function z = safeDivide(~, a, b)
            z = zeros(size(a));
            good = abs(b) > eps;
            z(good) = a(good) ./ b(good);
        end

        function V = precisionToVariance(~, precision)
            if precision == 0
                V = Inf;
            else
                V = 1 ./ precision;
            end
        end

        function X = sampleScalarNormal(~, mu, variance, sz, low, high)
            if isscalar(mu)
                mu = mu .* ones(sz);
            end
            if isscalar(variance)
                variance = variance .* ones(sz);
            end
            X = mu;
            stochastic = variance > 0 & isfinite(variance);
            X(stochastic) = mu(stochastic) + sqrt(variance(stochastic)) .* randn(sum(stochastic,'all'),1);
            X = min(max(X, low), high - eps);
        end

        function x = sampleBivariateTruncated(~, mu, covar)
            if any(~isfinite(covar), 'all') || any(~isfinite(mu))
                x = [min(max(mu(1), 0), 1-eps); mu(2)];
                return;
            end
            covar = (covar + covar') ./ 2 + 1e-12 .* eye(2);
            [L, flag] = chol(covar, 'lower');
            if flag ~= 0
                x = [min(max(mu(1), 0), 1-eps); mu(2)];
                return;
            end
            for attempt = 1:50
                candidate = mu + L * randn(2,1);
                if candidate(1) >= 0 && candidate(1) < 1
                    x = candidate;
                    return;
                end
            end
            x = [min(max(mu(1), 0), 1-eps); mu(2)];
        end

        function b = betaSample(obj, a, bpar)
            x = obj.gammaSample(a);
            y = obj.gammaSample(bpar);
            denom = x + y;
            b = zeros(size(denom));
            good = denom > 0;
            b(good) = x(good) ./ denom(good);
            b(~good) = 1;
        end

        function x = dirichletSample(obj, alpha)
            alpha = alpha(:);
            draws = obj.gammaSample(alpha);
            if sum(draws) <= 0
                draws = zeros(size(alpha));
                first = find(alpha > 0, 1);
                if isempty(first)
                    first = 1;
                end
                draws(first) = 1;
            end
            x = draws ./ sum(draws);
        end

        function g = gammaSample(~, shape)
            g = zeros(size(shape));
            good = shape > 0;
            if any(good, 'all')
                g(good) = randg(shape(good));
            end
        end

        function n = binomialSample(~, trials, prob)
            if trials <= 0 || prob <= 0
                n = 0;
            elseif prob >= 1
                n = trials;
            else
                n = sum(rand(1, trials) < prob);
            end
        end

        function Ainv = safeInverse(~, A)
            if rcond(A) < 1e-12
                Ainv = pinv(A);
            else
                Ainv = inv(A);
            end
        end

    end

    methods (Static)
        function indices = systematic_resampling(w)
            w = w(:)';
            w(~isfinite(w)) = 0;
            if sum(w) <= 0
                w = ones(size(w)) ./ numel(w);
            else
                w = w ./ sum(w);
            end
            N = numel(w);
            positions = ((0:N-1) + rand) ./ N;
            cumulative = cumsum(w);
            cumulative(end) = 1;
            indices = zeros(1,N);
            j = 1;
            for i = 1:N
                while positions(i) > cumulative(j)
                    j = j + 1;
                end
                indices(i) = j;
            end
        end

        function p = normal_pdf(x, m, v)
            if isscalar(v) && v <= 0
                p = zeros(size(x));
                p(abs(x - m) <= sqrt(eps)) = 1 ./ sqrt(eps);
                return;
            end
            v = max(v, eps);
            p = exp(-0.5 .* ((x - m).^2) ./ v) ./ sqrt(2*pi*v);
            p(~isfinite(p)) = realmax;
        end

        function p = normal_cdf(x, m, v)
            if isscalar(v) && v <= 0
                p = double(x >= m);
                return;
            end
            v = max(v, eps);
            p = 0.5 .* erfc(-(x - m) ./ sqrt(2 .* v));
            p = min(max(p, 0), 1);
        end

        function l = log_sum_exp(logP, dim)
            if nargin < 2
                dim = 1;
            end
            m = max(logP, [], dim);
            l = m + log(sum(exp(logP - m), dim));
            l(~isfinite(m)) = -Inf;
        end

        function p = stationary_distribution(T)
            c = size(T,1);
            A = T' - eye(c);
            b = zeros(c,1);
            A(end+1,:) = 1;
            b(end+1) = 1;
            x = A \ b;
            x(x < 0) = 0;
            if sum(x) == 0
                p = ones(1,c) ./ c;
            else
                p = x' ./ sum(x);
            end
        end

        function m = sample_num_tables(base, counts)
            m = zeros(size(counts));
            for i = 1:numel(counts)
                n = counts(i);
                b = base(i);
                if n <= 0 || b <= 0
                    continue;
                end
                tables = 0;
                for customer = 1:n
                    tables = tables + (rand < b ./ (b + customer - 1));
                end
                m(i) = tables;
            end
        end
    end
end
