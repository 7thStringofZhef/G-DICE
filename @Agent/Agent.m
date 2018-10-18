classdef Agent < handle
    properties
        idx; %unique agent idx
        curBeliefNode; %current location of the agent. Must be BeliefNode type!
        packageDelta = []; %[] if no package held
        packagePsi = []; %[] if no package held
        curTMAIdx; %[] if no TMA being conducted right now
        curTMACountdown = 1; %timer of TMA counts down until it's complete
        policy; %policy controller
        curPolicyNodeIdx; %agent's position in the policy controller graph
    end
    
    methods
        function obj = Agent(idx, initialBeliefNode, initTMAIdx)
            obj.idx = idx;
            obj.curBeliefNode = initialBeliefNode;
            obj.curTMAIdx = initTMAIdx;
        end
        
        %         function TMACountdownDecrement(obj)
        %             obj.curTMACountdown = obj.curTMACountdown - 1;
        %         end
        
        function updateBeliefNode(obj, beliefNodes)
            %if curTMACountdown is <=-1, it means the TMA failed, so beliefnode should NOT update
            if (obj.curTMACountdown > -1)
                switch obj.curTMAIdx
                    case 1
                        obj.curBeliefNode = beliefNodes(1);
                    case 2
                        obj.curBeliefNode = beliefNodes(2);
                    case 3
                        obj.curBeliefNode = beliefNodes(3);
                    case 4
                        obj.curBeliefNode = beliefNodes(4);
                    case 5
                        obj.curBeliefNode = beliefNodes(5);
                    case 6
                        obj.curBeliefNode = beliefNodes(4);
                    case 7
                        obj.curBeliefNode = beliefNodes(5);
                    otherwise
                        obj.curBeliefNode = obj.curBeliefNode;
                end
            end
        end
        
        function setNodeXe(obj, idxTMA, hasArrivedOrIsComplete, hasLeftOrJustStarted, isOutputOn)
            if (hasArrivedOrIsComplete == hasLeftOrJustStarted)
                error('hasArrived cannot be equal to hasLeft, since we do not allow zero-time TMAs.')
            else
                %for now, only change xE at node B1 or B2. At all other nodes, xE doesn't really matter (even for rendezvous since we're somewhat ignoring it in this case).
                if (obj.curBeliefNode.idx == 1 || obj.curBeliefNode.idx == 2)
                    if (idxTMA == 8 || idxTMA == 9)
                        if (hasLeftOrJustStarted)
                            %pickup package, so generate a new package for next person
                            obj.curBeliefNode.xeSamplePackage();
                            %also since you picked up a package, that means you're unavailable to help other agents
                            obj.curBeliefNode.xeSetPhi(obj.idx, 0, 1);
                            if (isOutputOn)
                                fprintf('Agent %i removed itself from the current node', obj.idx)
                            end
                        end
                    else
                        %TMA is unrelated to packages, and has to do with agent arriving or leaving
                        if (hasArrivedOrIsComplete)
                            %just arrived at B1 or B2
                            obj.curBeliefNode.xeSetPhi(obj.idx, 1, 0);
                        elseif (hasLeftOrJustStarted)
                            %just left B1 or B2
                            obj.curBeliefNode.xeSetPhi(obj.idx, 0, 1);
                        end
                    end
                else
                    error('You should not be updating XE from belief node %i (%s)', obj.curBeliefNode.idx, obj.curBeliefNode.name)
                end
            end
        end
        
        function nextTMAIdx = executePolicy(obj, curXeIdx)
            %move agent to next policy controller node, and assign next TMA idx
            [obj.curPolicyNodeIdx, nextTMAIdx] = obj.policy.getNextTMAIdx(obj.curPolicyNodeIdx, curXeIdx);
        end
        
        function setPolicy(obj, policy)
            if (isa(policy,'Policy') || isa(policy,'GraphPolicyController'))
                obj.policy = policy;
            else
                error('Policy must be of class type "Policy" or "GraphPolicyController"!')
            end
        end
    end
end