classdef RealTimeCOINEnsemble < handle
    %REALTIMECOINENSEMBLE  STUB interface for the multi-run averaging wrapper.
    %
    %   *** THIS IS A SIGNATURE-ONLY STUB. ***
    %   It is committed as the "blind base" so that test/validation authors can
    %   write their harnesses against the public interface WITHOUT seeing the
    %   real averaging or RNG implementation. Every query returns a
    %   correctly-shaped NaN placeholder and NO real logic is present. Author
    %   tests against docs/SPEC_ensemble.md (the contract), never against the
    %   values this stub returns.
    %
    %   The real implementation replaces the method bodies below (and may split
    %   them into per-method files under @RealTimeCOINEnsemble/). The signatures
    %   and shapes here are the contract; keep them stable.
    %
    %   See also docs/SPEC_ensemble.md, RealTimeCOIN.

    properties (SetAccess = private)
        % Number of independent member filters (runs).
        runs (1,1) double {mustBeInteger, mustBePositive} = 1;
        % Base RNG seed for the whole ensemble.
        seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0;
        % Worker cap: 0 => serial executor; >0 => parallel (parfor) capped here.
        max_cores (1,1) double {mustBeInteger, mustBeNonnegative} = 0;
        % Live-path parallel batch size (trials per parfor dispatch). Scheduling
        % only; must not affect numerical results.
        segment_length (1,1) double {mustBeInteger, mustBePositive} = 1;
        % Run weights (uniform in this version).
        weights double = 1;
    end

    properties (Access = private)
        % Name/value pairs forwarded verbatim to each member RealTimeCOIN.
        member_params = {};
        % Cached member state dimension (for correctly-shaped stub returns).
        state_dim_ (1,1) double = 1;
        % Trial counter (advances in lockstep with members).
        trial_ (1,1) double = 0;
    end

    properties (Dependent)
        % Common trial counter; equals every member's Trial.
        Trial;
    end

    methods
        function obj = RealTimeCOINEnsemble(varargin)
            %REALTIMECOINENSEMBLE Construct an ensemble (STUB: stores config only).
            if mod(numel(varargin), 2) ~= 0
                error('RealTimeCOINEnsemble:NameValuePairs', ...
                    'Arguments must be name/value pairs.');
            end
            mp = {};
            for k = 1:2:numel(varargin)
                name = varargin{k};
                val = varargin{k + 1};
                switch name
                    case 'runs',           obj.runs = val;
                    case 'seed',           obj.seed = val;
                    case 'max_cores',      obj.max_cores = val;
                    case 'segment_length', obj.segment_length = val;
                    otherwise
                        mp(end + 1 : end + 2) = {name, val};
                        if strcmp(name, 'state_dim')
                            obj.state_dim_ = val;
                        end
                end
            end
            obj.member_params = mp;
            obj.weights = ones(1, obj.runs) ./ obj.runs;
        end

        function val = get.Trial(obj)
            val = obj.trial_;
        end

        function observe_q(~, ~)
            %OBSERVE_Q Stage cue for all members (STUB: no-op).
            % (Real impl forwards q to every member; the stub does nothing.)
        end

        function observe_y(obj, ~)
            %OBSERVE_Y Feed feedback to all members (STUB: advance trial only).
            obj.trial_ = obj.trial_ + 1;
        end

        function u = motor_output(obj)
            %MOTOR_OUTPUT Averaged motor output (STUB: NaN of correct shape).
            u = nan(obj.state_dim_, 1);
        end

        function [mu, v] = state_moments(obj)
            %STATE_MOMENTS Averaged pooled-mixture moments (STUB: NaN).
            N = obj.state_dim_;
            mu = nan(N, 1);
            if N == 1
                v = NaN;
            else
                v = nan(N, N);
            end
        end

        function d = state_probability(obj, values)
            %STATE_PROBABILITY Averaged state density (STUB: NaN row).
            d = stubDensity(obj, values);
        end

        function d = state_feedback_probability(obj, values)
            %STATE_FEEDBACK_PROBABILITY Averaged feedback density (STUB: NaN row).
            d = stubDensity(obj, values);
        end

        function d = novel_state_probability(obj, values)
            %NOVEL_STATE_PROBABILITY Averaged novel-state density (STUB: NaN row).
            d = stubDensity(obj, values);
        end

        function d = novel_state_feedback_probability(obj, values)
            %NOVEL_STATE_FEEDBACK_PROBABILITY Averaged novel feedback (STUB: NaN row).
            d = stubDensity(obj, values);
        end

        function traces = simulate(obj, qSeq, ySeq)
            %SIMULATE Batch replay across runs (STUB: NaN traces of correct shape).
            if nargin < 3
                ySeq = [];
            end
            T = size(ySeq, 2);
            if T == 0 && nargin >= 2 && ~isempty(qSeq)
                T = numel(qSeq);
            end
            N = obj.state_dim_;
            traces = struct();
            traces.motor_output = nan(N, T);
            traces.state_mean = nan(N, T);
            if N == 1
                traces.state_var = nan(1, T);
            else
                traces.state_var = nan(N, N, T);
            end
            traces.Trial = 1:T;
        end
    end

    methods (Access = private)
        function d = stubDensity(obj, values)
            if obj.state_dim_ == 1
                K = numel(values);
            else
                K = size(values, 2);
            end
            d = nan(1, K);
        end
    end
end
