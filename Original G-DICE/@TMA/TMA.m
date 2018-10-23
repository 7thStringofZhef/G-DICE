classdef TMA < handle
    properties
        idx; %unique TMA index
        name; %a string descriptor of the TMA
        tau; %TMA completion time (also indicates initiation set, NaN for belief nodes where this TMA can't be initiated from)
        bTerm; %TMA termination belief. If NaN, indicates that termination is same as initiaton.
        r; %TMA reward
        allowableChildTMAIdxs; %indices of TMAs whose initiation sets match this one's termination set
        tauStDevParam = 0.3; %normalized stdev of tau. For generating tau distribution
    end
    methods
        function obj = TMA(idx, name, tau, bTerm, r, allowableChildTMAIdxs)
            obj.idx = idx;
            obj.name = name;
            obj.tau = tau;
            obj.bTerm = bTerm;
            obj.r = r;
            obj.allowableChildTMAIdxs = allowableChildTMAIdxs;
        end
        
        function sample = sampleTau(obj, currentBeliefNode)
            if (obj.idx == 13)
                %wait action is non-stochastic
                sample = obj.tau(currentBeliefNode.idx);
            else
                sample = obj.tau(currentBeliefNode.idx)*(1+ obj.tauStDevParam*randn());
                
                if (sample <0.5 && obj.tau(currentBeliefNode.idx)~=0)
                    %after rounding sample would have been 0
                    sample = obj.tau(currentBeliefNode.idx);
                end
                
                if isnan(sample)
%                     error('Tried to sample from a NaN tau! Are you sure you can call TMA #%i (%s) from the passed-in belief node (%s)?', obj.idx, obj.name, currentBeliefNode.name)
                    %usually this is an error, but for graph policy controller we do checking of valid TMAs in the
                    %Domain class. So for now, just set tau = 0
                    sample = 0;
                else
                    sample = round(sample);
                end
            end
        end
    end
end