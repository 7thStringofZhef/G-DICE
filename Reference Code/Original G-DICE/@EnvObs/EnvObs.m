classdef EnvObs < handle
    properties
        psi; %package size
        allPsis; %set of all psis
        delta; %package delivery destination
        allDeltas; %set of all deltas
        phi; %array of indices of agents at the current belief node
    end
    methods
        function obj = EnvObs(allPsis, allDeltas)
            obj.allPsis = allPsis;
            obj.allDeltas = allDeltas;
        end
        
        %you must ALWAYS call this to get xe, because it does automatic checking of agent indices to report whether or
        %not ANOTHER agent exists in the current belief node
        function xeIdx = getXeIdx(obj, idxCallerAgent, callerAgentPackageDelta, curTMAIdx, isOutputOn)
            
            if (isempty(obj.psi) && isempty(obj.delta) && isempty(obj.phi))
                xe = [0 0 0];
            elseif (isempty(obj.phi(obj.phi ~= idxCallerAgent)))
                %if no other agents other than yourself exist at the current node
                xe = [obj.psi obj.delta 0];
            else
                %another agent is there with you
                xe = [obj.psi obj.delta 1];
            end
            
            if (isOutputOn)
                fprintf('Agent %i called getXeIdx (TMA = %i, package delta = %i), xe = %s, psi = %i, delta = %i, phi = %s \n', idxCallerAgent, curTMAIdx, callerAgentPackageDelta, num2str(xe), obj.psi, obj.delta, num2str(obj.phi))
            end
            %-------------------------------------------------------------%
            %for pickup tasks, assign xeIdx based on package delivery destination.
            %if agent was randomly assigned pickup task by policyController (even though invalid and no callerAgentPackageDelta exists)
            %then this code will have no impact on xeIdx generation
            if (curTMAIdx == 8)
                if (callerAgentPackageDelta == 1)
                    xeIdx = 2;
                    return
                elseif (callerAgentPackageDelta == 2)
                    xeIdx = 3;
                    return
                elseif (callerAgentPackageDelta == 3)
                    xeIdx = 4;
                    return
                end
            elseif (curTMAIdx == 9)
                if (callerAgentPackageDelta == 1)
                    xeIdx = 11;
                    return
                elseif (callerAgentPackageDelta == 2)
                    xeIdx = 12;
                    return
                end
            end
            %-------------------------------------------------------------%
            
            if isequal(xe, [0 0 0])
                %null case
                xeIdx = 1;
            elseif isequal(xe, [1 1 0])
                %pkg = small, dest = d1, other agent = absent
                xeIdx = 2;
            elseif isequal(xe, [1 2 0])
                %pkg = small, dest = d2, other agent = absent
                xeIdx = 3;
            elseif isequal(xe, [1 3 0])
                %pkg = small, dest = r, other agent = absent
                xeIdx = 4;
            elseif isequal(xe, [2 1 0])
                %pkg = large, dest = d1, other agent = absent
                xeIdx = 5;
            elseif isequal(xe, [2 2 0])
                %pkg = large, dest = d2, other agent = absent
                xeIdx = 6;
            elseif isequal(xe, [2 3 0])
                %pkg = large, dest = r, other agent = absent
                xeIdx = 7;
            elseif isequal(xe, [1 1 1])
                %pkg = small, dest = d1, other agent = present
                xeIdx = 8;
            elseif isequal(xe, [1 2 1])
                %pkg = small, dest = d2, other agent = present
                xeIdx = 9;
            elseif isequal(xe, [1 3 1])
                %pkg = small, dest = r, other agent = present
                xeIdx = 10;
            elseif isequal(xe, [2 1 1])
                %pkg = large, dest = d1, other agent = present
                xeIdx = 11;
            elseif isequal(xe, [2 2 1])
                %pkg = large, dest = d2, other agent = present
                xeIdx = 12;
            elseif isequal(xe, [2 3 1])
                %pkg = large, dest = r, other agent = present
                xeIdx = 13;
            else
                error('Received xe: [%i %i %i], which is impossible!', xe)
            end
        end
        
        function setXeNull(obj)
            %should be used at belief nodes where xe doesn't come into play
            obj.psi = 0;
            obj.delta = 0;
            obj.phi = 0;
        end
        
        %@B1 and B2 - call setPhi after an agent COMPLETES arrival task, and after an agent STARTS to leave,
        %samplePackage when agent STARTS pickup of a package
        %@R - nothing for now, acts like a default delivery destination
        %@D1 and D2 - nothing
        
        %sampling should only be called when xE needs to explicitly changed
        function samplePackage(obj)
            obj.delta = datasample(obj.allDeltas,1);
            if (obj.delta == 3)
                %going to rendezvous location, only small pkg allowed
                obj.psi = 1;
            else
                obj.psi = datasample(obj.allPsis,1);
            end
        end
        
        function setPhi(obj, idxAgent, hasArrived, hasLeft)
            if (hasArrived == hasLeft)
                error('Agent %i cannot simultaneously arrive at and leave a node.')
            else
                %always remove the agent's index, just in case it gets double added
                obj.phi = obj.phi(obj.phi ~= idxAgent);                
                
                %add it back if agent has arrived.
                if (hasArrived)
                    obj.phi = [obj.phi; idxAgent];
                end
            end
            
        end
    end
end