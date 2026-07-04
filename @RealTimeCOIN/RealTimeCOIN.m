classdef RealTimeCOIN < handle
    %REALTIMECOIN Sequential scalar COIN particle filter.
    %
    %   The core API is state-machine like: observe_q(q) records the cue for the
    %   next trial, observe_y(y) processes the trial feedback, and query methods
    %   expose the current posterior/predictive summaries. Internally, particles
    %   are stored in vectorized arrays following the original COIN.m
    %   implementation.
    %
    %   Query methods mirror every quantity COIN.m is able to plot, evaluated at
    %   the current trial. Distributions/densities: state_probability,
    %   state_given_context_probability, novel_state_probability,
    %   state_feedback_probability, state_feedback_given_context_probability,
    %   novel_state_feedback_probability, retention_given_context_probability,
    %   drift_given_context_probability, bias_given_context_probability,
    %   bias_probability, local_transition_probabilities, local_cue_probabilities,
    %   global_transition_probabilities, global_cue_probabilities,
    %   stationary_context_probabilities. Probability vectors:
    %   predicted_context_probabilities, responsibilities (and *_local variants),
    %   sampled_context_count. Scalar summaries: motor_output, state_moments,
    %   explicit_component, implicit_component, and the c* traces
    %   state_cstar1/2/3, predicted_probability_cstar1/3, kalman_gain_cstar1/2.
    %   Retention/drift/bias densities and scalar Kalman gains are scalar-model
    %   only (state_dim == 1).

    properties
        num_particles (1,1) double {mustBeInteger,mustBePositive} = 100;
        max_contexts (1,1) double {mustBeInteger,mustBePositive} = 10;

        % Dimensionality of the latent state (and, since the observation map
        % is the identity, of the observation as well). state_dim == 1 selects
        % the original scalar pipeline verbatim; state_dim > 1 selects the
        % multi-dimensional pipeline. See observe_y.m for the dispatch.
        state_dim (1,1) double {mustBeInteger,mustBePositive} = 1;

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

        % Optional explicit noise covariances for the multi-dimensional model.
        % Leave empty ([]) to use the isotropic defaults derived from the
        % scalar sigma_* properties: Q = sigma_process_noise^2 * I and
        % R = (sigma_sensory_noise^2 + sigma_motor_noise^2) * I. When supplied
        % they must be symmetric positive-semidefinite state_dim-by-state_dim
        % matrices (validated at construction). Ignored when state_dim == 1.
        process_noise_covariance double = [];
        observation_noise_covariance double = [];

        infer_bias (1,1) logical = false;
    end

    properties (Access = private)
        D;
        pending_q = [];
        trial = 0;
        cue_values = [];
        alignment_cache = [];
        alignment_seed = [];
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
            validateMultiDimConfig(obj);
            resetParticles(obj);
        end

        function val = get.Trial(obj)
            val = obj.trial;
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
