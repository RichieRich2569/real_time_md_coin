classdef RealTimeCOINEnsemble < handle
    %REALTIMECOINENSEMBLE Multi-run averaging wrapper over RealTimeCOIN.
    %
    %   An ensemble orchestrates R independent RealTimeCOIN filters ("members" /
    %   "runs") that all consume the IDENTICAL observation stream, fed one trial
    %   at a time. Query methods return the equal-weight average across runs of
    %   the corresponding single-model quantity -- i.e. the readout of the pooled
    %   mixture distribution that gives every run weight 1/R. This is the
    %   real-time analogue of COIN.m's offline "runs", used for Monte-Carlo
    %   variance reduction ("probability averaging") and validation.
    %
    %   The wrapper does NOT change any RealTimeCOIN per-trial behaviour; each
    %   member is an ordinary RealTimeCOIN object. Randomness is made independent
    %   AND reproducible by giving member k a dedicated RandStream substream
    %   indexed by k (Threefry, NumStreams = runs, Seed = seed); the global
    %   default stream is swapped to member k's stream around member k's steps and
    %   restored afterwards, so the unchanged RealTimeCOIN modules draw from it
    %   without any edits. Results are therefore independent of the executor
    %   (serial vs parfor) and of segment_length.
    %
    %   Construction (name/value): ensemble parameters runs, seed, max_cores,
    %   segment_length; every other name/value pair is forwarded verbatim and
    %   identically to each member RealTimeCOIN constructor.
    %
    %   API: observe_q, observe_y (state machine, mirror RealTimeCOIN); averaged
    %   queries motor_output, state_moments, state_probability,
    %   state_feedback_probability, novel_state_probability,
    %   novel_state_feedback_probability; batch replay simulate.
    %
    %   See also docs/SPEC_ensemble.md, RealTimeCOIN.

    properties (SetAccess = private)
        % Number of independent member filters (runs).
        runs (1,1) double {mustBeInteger, mustBePositive} = 1;
        % Base RNG seed for the whole ensemble.
        seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0;
        % Worker cap: 0 => serial executor; >0 => parallel (parfor) capped here.
        % Affects only how simulate() schedules work, never the numerical result.
        max_cores (1,1) double {mustBeInteger, mustBeNonnegative} = 0;
        % Reserved live-path parallel batch size (trials per dispatch). A
        % scheduling hint only; it must never affect numerical results.
        segment_length (1,1) double {mustBeInteger, mustBePositive} = 1;
        % Run weights (uniform in this version).
        weights double = 1;
    end

    properties (Access = private)
        % 1xR cell of member RealTimeCOIN objects (independent handles).
        members = {};
        % 1xR cell of per-member RandStream substreams.
        streams = {};
        % Name/value pairs forwarded verbatim to each member RealTimeCOIN.
        member_params = {};
        % Member state dimension (cached from members{1} for shaping outputs).
        state_dim_ (1,1) double = 1;
        % Trial counter (advances in lockstep with members).
        trial_ (1,1) double = 0;
        % RNG generator name for the per-member substreams.
        rng_generator = 'threefry';
    end

    properties (Dependent)
        % Common trial counter; equals every member's Trial.
        Trial;
    end

    methods
        function obj = RealTimeCOINEnsemble(varargin)
            %REALTIMECOINENSEMBLE Construct an ensemble of R seeded members.
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
                end
            end
            obj.member_params = mp;
            obj.weights = ones(1, obj.runs) ./ obj.runs;

            % Build members, each under its own reproducible substream so that
            % construction randomness (resetParticles) also belongs to member k.
            prev = RandStream.getGlobalStream();
            restore = onCleanup(@() RandStream.setGlobalStream(prev));
            obj.streams = cell(1, obj.runs);
            obj.members = cell(1, obj.runs);
            for k = 1:obj.runs
                obj.streams{k} = makeMemberStream(obj.rng_generator, obj.runs, k, obj.seed);
                RandStream.setGlobalStream(obj.streams{k});
                obj.members{k} = RealTimeCOIN(obj.member_params{:});
            end
            obj.state_dim_ = obj.members{1}.state_dim;
        end

        function val = get.Trial(obj)
            val = obj.trial_;
        end
    end
end
