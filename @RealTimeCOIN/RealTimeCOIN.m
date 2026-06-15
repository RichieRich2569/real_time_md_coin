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
            resetParticles(obj);
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

            q_val = consumePendingCue(obj);
            predictContext(obj, q_val);
            predictStates(obj);
            predictStateFeedback(obj);
            resampleParticles(obj, y_val, q_val);
            sampleContext(obj, q_val);
            updateBeliefAboutStates(obj, y_val);
            sampleStates(obj, y_val);
            updateSufficientStatistics(obj, y_val, q_val);
            sampleParameters(obj);

            obj.trial = obj.trial + 1;
            invalidateContextAlignment(obj);
        end

        function probs = context_responsibilities(obj)
            weights = contextProbabilityVector(obj, "responsibilities");
            probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
            for c = 1:numel(weights)
                if weights(c) > 0
                    probs(c) = weights(c);
                end
            end
        end

        function probs = context_predicted_probabilities(obj)
            weights = contextProbabilityVector(obj, "predicted");
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
            alignment = ensureContextAlignment(obj);
            M = alignment.global_contexts.state_mean;
            V = alignment.global_contexts.state_var;
            active = activeSummaryContexts(obj);
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
            resetParticles(obj);
            obj.pending_q = [];
            obj.trial = 0;
            obj.cue_values = [];
        end

        function saveModel(obj, filename, setStationary)
            if nargin < 3
                setStationary = true;
            end
            if setStationary
                saved = serializableState(obj);
                obj.set_stationary();
                model = serializableState(obj);
                save(filename, 'model');
                restoreSerializableState(obj, saved);
            else
                model = serializableState(obj);
                save(filename, 'model');
            end
        end

        function loadModel(obj, filename)
            S = load(filename);
            if isfield(S, 'model')
                restoreSerializableState(obj, S.model);
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
            p = contextProbabilityVector(obj, "predicted");
        end

        function p = responsibilities(obj)
            p = contextProbabilityVector(obj, "responsibilities");
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
            n = contextProbabilityVector(obj, "count");
        end

        function S = diagnostics(obj)
            S = struct();
            alignment = ensureContextAlignment(obj);
            S.trial = obj.trial;
            S.C = alignment.K;
            S.context = globalSampledContexts(obj, alignment);
            S.predicted_probabilities = globalContextWeights(obj, obj.D.predicted_probabilities, alignment);
            S.responsibilities = globalContextWeights(obj, obj.D.responsibilities, alignment);
            S.state_mean = globalContextMatrix(obj, obj.D.state_mean, alignment);
            S.state_var = globalContextMatrix(obj, obj.D.state_var, alignment);
            S.state_feedback_mean = globalContextMatrix(obj, obj.D.state_feedback_mean, alignment);
            S.state_feedback_var = globalContextMatrix(obj, obj.D.state_feedback_var, alignment);
            S.retention = globalContextMatrix(obj, obj.D.retention, alignment);
            S.drift = globalContextMatrix(obj, obj.D.drift, alignment);
            S.bias = globalContextMatrix(obj, obj.D.bias, alignment);
            S.global_transition_probabilities = globalContextMatrix(obj, obj.D.global_transition_probabilities, alignment);
            S.local_transition_matrix = globalTransitionTensor(obj, obj.D.local_transition_matrix, alignment);
            S.global_cue_probabilities = obj.D.global_cue_probabilities(:, alignment.modal_particle_indices);
            S.local_cue_matrix = globalCueTensor(obj, obj.D.local_cue_matrix, alignment);
            S.alignment = alignment;
            S.raw = obj.D;
        end

        function p = predictive_state_feedback_cdf(obj, y, q)
            if nargin < 3
                q = obj.pending_q;
            end
            qLabel = peekCueLabel(obj, q);
            [W, M, V] = previewPredictiveFeedback(obj, qLabel);
            p = sum(W .* obj.normal_cdf(y, M, V), 'all') ./ obj.num_particles;
            p = min(max(p, 0), 1);
        end

        function p = predictive_cue_p_value(obj, q, u)
            if nargin < 3
                u = rand;
            end
            qLabel = peekCueLabel(obj, q);
            if isempty(qLabel)
                p = NaN;
                return;
            end
            [pmf, labels] = previewCuePmf(obj);
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
            qLabel = peekCueLabel(obj, q);
            [W, M, ~] = previewPredictiveFeedback(obj, qLabel);
            u = sum(W .* M, 'all') ./ obj.num_particles;
        end

        function alignment = context_alignment(obj)
            alignment = ensureContextAlignment(obj);
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
