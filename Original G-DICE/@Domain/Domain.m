classdef Domain < handle
    properties
        beliefNodes = [];
        TMAs = [];
        numTMAs = 0;
        numBeliefNodes = 5;
        agents = [];
        validTMATransitions;
    end
    methods
        function obj = Domain()
            %Map: Base1 (B1), Base2 (B2), Rendezvous (R), Delivery1 (D1), Delivery2 (D2)
            %0  0 D2 0
            %B1 0 0  R
            %0  0 0  0
            %B2 0 D1 0
            
            %-------------------------------------------------------------------------%
            %Belief Nodes
            psiTemp = [1; 2]; %for now ALWAYS have a package at the base, never without a package
            deltaTemp = [1; 2; 3]; %1 = D1, 2 = D2, 3 = R
            xETempB1 = EnvObs(psiTemp, deltaTemp);
            b1BeliefNode = obj.appendToBeliefNodes(1, 'B1', xETempB1);
            xETempB2 = EnvObs(psiTemp, deltaTemp);
            obj.appendToBeliefNodes(2, 'B2', xETempB2);
            
            psiTemp = [];
            deltaTemp = [];
            xETempR = EnvObs(psiTemp, deltaTemp);
            obj.appendToBeliefNodes(3, 'R', xETempR);
            
            psiTemp = [];
            deltaTemp = [];
            xETempD1 = EnvObs(psiTemp, deltaTemp);
            obj.appendToBeliefNodes(4, 'D1', xETempD1);
            xETempD2 = EnvObs(psiTemp, deltaTemp);
            obj.appendToBeliefNodes(5, 'D2', xETempD2);
            
            %-------------------------------------------------------------------------%
            %Agents - both start at B1 (TMA idx 1 == GOTO B1)
            tempAgent = Agent(1, b1BeliefNode, 1);
            obj.appendToAgents(tempAgent);
            tempAgent = Agent(2, b1BeliefNode, 1);
            obj.appendToAgents(tempAgent);
            
            %-------------------------------------------------------------------------%
            %Macro Actions
            
            %MA1 - go to B1
            %MA2 - go to B2
            %MA3- go to R
            %MA4 - go to D1
            %MA5 - go to D2
            
            idx = 1;
            name = 'Go to B1';
            tauTemp = [NaN; 2; 4; 3; 2];
            bTermTemp = [1; 0; 0; 0; 0];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [2;3;8;9];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 2;
            name = 'Go to B2';
            tauTemp = [2; NaN; 4; 2; 4];
            bTermTemp = [0; 1; 0; 0; 0];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [1;3;8;9];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 3;
            name = 'Go to R';
            tauTemp = [3; 4; NaN; NaN; NaN];
            bTermTemp = [0; 0; 1; 0; 0];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [1;2;12];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 4;
            name = 'Go to D1';
            tauTemp = [3; 2; NaN; NaN; NaN];
            bTermTemp = [0; 0; 0; 1; 0];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [1;2;10];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 5;
            name = 'Go to D2';
            tauTemp = [2; 4; NaN; NaN; NaN];
            bTermTemp = [0; 0; 0; 0; 1];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [1;2;10];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            %-------------------------------------------------------------------------%
            %MA6 - joint go to D1
            %MA7 - joint go to D2
            
            idx = 6;
            name = 'Joint go to D1';
            tauTemp = [3; 2; NaN; NaN; NaN]*1.5;
            bTermTemp = [0; 0; 0; 1; 0];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [11];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 7;
            name = 'Joint go to D2';
            tauTemp = [2; 4; NaN; NaN; NaN]*1.5;
            bTermTemp = [0; 0; 0; 0; 1];
            rTemp = zeros(5,1);
            allowableChildTMAsIdxsTemp = [11];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            %-------------------------------------------------------------------------%
            %MA8 - pick up package
            %MA9 - joint pick up package (takes longer)
            
            %initiation only at B1 and B2
            idx = 8;
            name = 'Pick up package';
            tauTemp = [1; 1; NaN; NaN; NaN];
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = [0; 0; NaN; NaN; NaN];
            allowableChildTMAsIdxsTemp = [3;4;5];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 9;
            name = 'Joint pick up package';
            tauTemp = [1.5; 1.5; NaN; NaN; NaN];
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = [0; 0; NaN; NaN; NaN];
            allowableChildTMAsIdxsTemp = [6;7];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            %-------------------------------------------------------------------------%
            %MA10 - put down package
            %MA11 - joint put down package (takes longer, higher reward)
            
            %initiation only at D1 and D2
            idx = 10;
            name = 'Put down package';
            tauTemp = [NaN; NaN; NaN; 1; 1];
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = [NaN; NaN; NaN; 1; 1];
            allowableChildTMAsIdxsTemp = [1;2];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            idx = 11;
            name = 'Joint put down package';
            tauTemp = [NaN; NaN; NaN; 1.5; 1.5];
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = [NaN; NaN; NaN; 3; 3];
            allowableChildTMAsIdxsTemp = [1;2];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            %-------------------------------------------------------------------------%
            %MA12 - place package on truck
            
            %initiation only at R
            idx = 12;
            name = 'Place package on truck';
            tauTemp = [NaN; NaN; 1; NaN; NaN];
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = [NaN; NaN; 0; NaN; NaN];
            allowableChildTMAsIdxsTemp = [1;2];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            %-------------------------------------------------------------------------%
            %MA13 - wait for 1 time unit
            %initiation anywhere
            
            idx = 13;
            name = 'Wait';
            tauTemp = 1*ones(5,1);
            bTermTemp = [NaN; NaN; NaN; NaN; NaN];
            rTemp = 0*ones(5,1);
            allowableChildTMAsIdxsTemp = [13];
            obj.appendToTMAs(idx, name, tauTemp, bTermTemp, rTemp, allowableChildTMAsIdxsTemp);
            
            obj.numTMAs = length(obj.TMAs);
            %-------------------------------------------------------------------------%
            %initialize environmental state
            obj.initXe();
                                       % [obj.psi obj.delta obj.phi] = [size, destination, other agents avail]
                                       %  1   2   3   4   5   6   7   8   9  10  11  12  13
                                       %000 110 120 130 210 220 230 111 121 131 211 221 231
            obj.validTMATransitions = [  -1   0   0   0   0   0   0   0   0   0   0   0   0; %1 -goto b1 (at b1)
                                         -1   0   0   0   0   0   0   0   0   0   0   0   0; %2 -goto b2 (at b2)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%3 -goto r (at r)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%4 -goto d1 (at d1)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%5 -goto d2 (at d2)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%6 -joint goto d1 (at d1)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%7 -joint goto d2 (at d2)
                                         -1   0   0   0  -1  -1  -1   0   0   0  -1  -1  -1;%8 -pickup pkg (at b1 or b2) %this is a special case where the observations MUST COME BEFOREA NEW PKG IS GENERATED!!
                                         -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0   0  -1;%9 -joint pickup pkg (at b1 or b2)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%10 -put down pkg (at d1 or d2)
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%11 -joint put down pkg (at d1 or d2) 
                                          0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%12 -place on truck (at r)
                                          0   0   0   0   0   0   0   0   0   0   0   0   0;%13 -wait (anywhere)
                                      ];
                                  

        end
        
        function initXe(obj)
            %set agents in xE
            for idxAgent = 1:length(obj.agents)
                obj.agents(idxAgent).curBeliefNode.xeSetPhi(idxAgent, 1, 0)
            end
            
            %init packages at bases
            for idxBase = 1:2
                obj.beliefNodes(idxBase).xeSamplePackage();
            end
        end
        
        function appendToTMAs(obj, idx, name, tau, bTerm, r, allowableChildTMAIdxs)
            newTMA = TMA(idx, name, tau, bTerm, r, allowableChildTMAIdxs);
            obj.TMAs = [obj.TMAs; newTMA];
        end
        
        function newBeliefNode = appendToBeliefNodes(obj, idx, name, xE)
            newBeliefNode = BeliefNode(idx, name, xE);
            obj.beliefNodes = [obj.beliefNodes; newBeliefNode];
        end
        
        function appendToAgents(obj, agent)
            obj.agents = [obj.agents; agent];
        end
        
        function setPolicyForAllAgents(obj, policyController)
            for idxAgent = 1:length(obj.agents)
                %set agent's policy
                obj.agents(idxAgent).setPolicy(policyController);
                %reset agent to be at first node of policy
                obj.agents(idxAgent).curPolicyNodeIdx = 1;
            end
        end
        
        function [value, completionTime] = evalCurPolicy(obj, isOutputOn, numPackagesGoal)
            value = 0;
            
            %time increments by units of 1
            if (nargin > 2)
                %run until package goal is hit or time runs out
                maxTime = 50; %used to be 50
            else
                %run until time runs out
                maxTime = 50; %used to be 50
                numPackagesGoal = 9999;
            end
            
            curTime = 0;
            numPackageDelivered = 0;
            
            gamma = 0.99;
            
            while ((curTime < maxTime) && (numPackageDelivered < numPackagesGoal))
                %first, check completion of TMAs and update xE accordingly
                for idxAgent = 1:length(obj.agents)
                    curAgent = obj.agents(idxAgent);
                    
                    %reduce TMA timer
%                     curAgent.TMACountdownDecrement();
                    curAgent.curTMACountdown = curAgent.curTMACountdown - 1;
                    
                    if (curAgent.curTMACountdown <=0)
                        %update agent's belief node based on the TMA it just completed. Inside here, we check if TMACountdown is <-100, it
                        %means the TMA failed, so agent belief node shouldn't update. 
                        curAgent.updateBeliefNode(obj.beliefNodes);
                        
                        if (curAgent.curBeliefNode.idx == 1 || curAgent.curBeliefNode.idx == 2)
                            %for GOTO B1 or B2 task is complete, need to set phi (availability at that base) as soon as arrival occurs
                            %no changes to xE's packages at that node are made, since the agent is just arriving there/COMPLETING its GOTO TMA.
                            %package changes are only done when the agent STARTS doing a pickup TMA (down below, at "if (newTMAIdx ~= 13)"), NOT here
                            curAgent.setNodeXe(curAgent.curTMAIdx, 1, 0, isOutputOn)
                            if (isOutputOn)
                                fprintf('Agent %i added to belief node %i', curAgent.idx, curAgent.curBeliefNode.idx)
                            end
                        end
                        
                        %package delivery complete, remove packageDelta from curAgent, give reward
                        %also, only get the reward if solo drop of small pkg, or joint drop of large pkg
                        if (~isempty(curAgent.packagePsi) && ((curAgent.curTMAIdx == 10 && curAgent.packagePsi == 1) || (curAgent.curTMAIdx == 11 && curAgent.packagePsi == 2)))
                            %make sure you delivered package to correct destination
                           if (curAgent.packageDelta == curAgent.curBeliefNode.idx-3)
                               
                                value = value + gamma^curTime*obj.TMAs(curAgent.curTMAIdx).r(curAgent.curBeliefNode.idx);
                                if (obj.TMAs(curAgent.curTMAIdx).r(curAgent.curBeliefNode.idx) > 2)
%                                     fprintf('delivered joint package%d!\n', curAgent.curTMAIdx)
%                                     pause
                                else 
%                                     fprintf('delivered single package%d!\n', curAgent.curTMAIdx)
                                end
                                numPackageDelivered  = numPackageDelivered + 1;
                           end
                           curAgent.packageDelta = []; 
                           curAgent.packagePsi = [];
                        end
                        
                        if (isOutputOn)
                            fprintf('\nAgent %i finished its TMA and is at belief node %s\n\n', curAgent.idx, curAgent.curBeliefNode.name);
                        end
                    end
                end
                
                jointPickupTMAChosen = 0;
                jointActionChosen = 0;
                fellowAgentTMAIdx = 0;
                fellowAgentTMATimer = 0;
                fellowAgentDelta = 0;
                fellowAgentPsi = 0;
                
                %now, do assignments of next TMAs based on new xE
                for idxAgent = 1:length(obj.agents)
                    curAgent = obj.agents(idxAgent);
                    %if TMA is complete
                    if (curAgent.curTMACountdown <= 0)
                        %pick a new TMA
                        %JOINT CODE ONLY WORKS FOR 2 AGENT CASE!
                        if (jointPickupTMAChosen)
                            newTMAIdx = 9;
                            curAgent.curTMACountdown = fellowAgentTMATimer;
                            curAgent.packageDelta = fellowAgentDelta;
                            curAgent.packagePsi = fellowAgentPsi;
                            if (isOutputOn)
                                fprintf('Agent %i forced to choose joint pickup TMA 9 (%s, timer = %f). It is currently at belief node %s\n', curAgent.idx, obj.TMAs(newTMAIdx).name, curAgent.curTMACountdown, curAgent.curBeliefNode.name)
                            end
                        elseif (jointActionChosen)
                            newTMAIdx = fellowAgentTMAIdx;
                            curAgent.curTMACountdown = fellowAgentTMATimer;
                            if (isOutputOn)
                                fprintf('Agent %i forced to choose joint TMA %i (%s, timer = %f). It is currently at belief node %s\n', curAgent.idx, newTMAIdx, obj.TMAs(newTMAIdx).name, curAgent.curTMACountdown, curAgent.curBeliefNode.name)
                            end
                        else
                            if (isOutputOn)
                                fprintf('Agent %i at beliefNode %s will choose its TMA now\n', curAgent.idx, curAgent.curBeliefNode.name)
                            end
                                %replace getNextTMAIdx in graph-based case                            
                                newTMAIdx = curAgent.executePolicy(curAgent.curBeliefNode.xE.getXeIdx(idxAgent, curAgent.packageDelta, curAgent.curTMAIdx, isOutputOn));
                            
                            %set timer for agent
                            curAgent.curTMACountdown = obj.TMAs(newTMAIdx).sampleTau(curAgent.curBeliefNode);
                            if (isOutputOn)
                                fprintf('Agent %i sees xE %i and chooses TMA %i (%s, timer = %f). It is currently at belief node %s\n', curAgent.idx, curAgent.curBeliefNode.xE.getXeIdx(idxAgent, curAgent.packageDelta, curAgent.curTMAIdx, isOutputOn), newTMAIdx, obj.TMAs(newTMAIdx).name, curAgent.curTMACountdown, curAgent.curBeliefNode.name)
                            end
                            
                            %for pickup TMAs, save delivery location for next turn
                            if (newTMAIdx == 8 || newTMAIdx == 9)
                                %TODO - in graph case, TMA is chosen randomly, so you might have chosen pickup action even when you're at a delivery destination etc.
                                %make sure in this case that curBeliefNode.xE.delta is NOT null
                                curAgent.packageDelta = curAgent.curBeliefNode.xE.delta;
                                curAgent.packagePsi = curAgent.curBeliefNode.xE.psi;
                            end
                        end
                        
                        %for GOTO *somewhere* tasks, need to remove self from phi (agent availability) of current node as soon as departure occurs
                        %this also generates a new package for idxTMA == 8 and idxTMA == 9 (pickup TMAs), automatically
                        if (newTMAIdx ~= 13)
                            if (curAgent.curBeliefNode.idx == 1 || curAgent.curBeliefNode.idx == 2)
                                %if you try to joint pickup a small package, or solo pickup a large package, the TMA
                                %should fail (set countdown to 0), and the package should NOT be resampled (so don't
                                %touch xE, don't remove yourself from phi either
                                if ((curAgent.curBeliefNode.xE.psi == 1 && newTMAIdx == 9) || (curAgent.curBeliefNode.xE.psi == 2 && newTMAIdx == 8))
                                    curAgent.curTMACountdown = -100;
                                else
                                    curAgent.setNodeXe(curAgent.curTMAIdx, 0, 1, isOutputOn)
                                end
                                
                                if (isOutputOn)
                                    fprintf('Agent %i removed itself from the current node\n', curAgent.idx)
                                end
                            end
                        end
                        
                        %for joint pickup tasks, need to coordinate with other agent and also set new pkg at current base
                        if (newTMAIdx == 9)
%                             sprintf('Joint pickup TMA 9 chosen!')
%                             pause
                            if (jointPickupTMAChosen)
                                %your other agent already started joint pickup action, and instructed you to do the same
                                %so reset the pkg status at the base
                                if (curAgent.curBeliefNode == 1 || curAgent.curBeliefNode == 2)
                                    curAgent.setNodeXe(newTMAIdx, 0, 1, isOutputOn)
                                end
                                jointPickupTMAChosen = 0;
                                fellowAgentTMATimer = 0;
                                fellowAgentDelta = 0;
                                fellowAgentPsi = 0;
                            else
                                %you're the first agent to choose the pickup action, so throw a flag telling your friend
                                %to help you
                                jointPickupTMAChosen = 1;
                                fellowAgentTMATimer = curAgent.curTMACountdown;
                                fellowAgentDelta = curAgent.packageDelta;
                                fellowAgentPsi = curAgent.packagePsi;
                            end
                        %joint goto d1/d2, or joint put down
                        elseif (newTMAIdx == 6 || newTMAIdx == 7 || newTMAIdx == 11)
                            if (jointActionChosen)
                                jointActionChosen = 0;
                                fellowAgentTMATimer = 0;
                                fellowAgentTMAIdx = 0;
                            else
                                jointActionChosen = 1;
                                fellowAgentTMATimer = curAgent.curTMACountdown;
                                fellowAgentTMAIdx = newTMAIdx;
                            end
                        end
                        
                        %if new TMA is not a valid child of previous TMA, set curAgent.curTMACountdown to 0 so that the
                        %TMA is skipped on next turn
                        if ((newTMAIdx) ~= obj.TMAs(curAgent.curTMAIdx).allowableChildTMAIdxs)
                           curAgent.curTMACountdown = -100;
%                            fprintf('Agent %d: %s -> %s at beliefnode %s, which caused it to fail\n', curAgent.idx, obj.TMAs(curAgent.curTMAIdx).name, obj.TMAs(newTMAIdx).name, curAgent.curBeliefNode.name);
%                         else
%                             curAgent.curTMAIdx = newTMAIdx;    
                        end
                        
                        curAgent.curTMAIdx = newTMAIdx;    
%                         pause
                        
                    end
                end
                
                curTime = curTime + 1;
            end
            
            %not a single package delivered - policy fails
            if (numPackageDelivered < numPackagesGoal)
                completionTime = -1;
            else
                completionTime = curTime;
            end
        end
        
        function printAllowableTMAs(obj)
            for idxTMA = 1:obj.numTMAs
                idxTMA
                obj.TMAs(idxTMA).allowableChildTMAIdxs
            end
        end
        
        function constructTree(obj)
            for idxTMA = 1:obj.numTMAs
                %if termination set contains NaN, then agent's termination set is equal to its initiation set
                if isnan(obj.TMAs(idxTMA).bTerm)
                    %find all initiation set elements that are allowable (or, are not equal to NaN)
                    allTerminationsIdx = find(isnan(obj.TMAs(idxTMA).tau)==0);
                    for idxTermination = allTerminationsIdx
                        obj.TMAs(idxTMA).allowableChildTMAIdxs = findAllowableTMAs(obj, idxTermination);
                    end
                else
                    %there is a specific termination set associated with the TMA
                    for idxTermination = 1:obj.numBeliefNodes
                        if (obj.TMAs(idxTMA).bTerm(idxTermination)==1)
                            obj.TMAs(idxTMA).allowableChildTMAIdxs = findAllowableTMAs(obj, idxTermination);
                        end
                    end
                end
            end
        end
        
        function allowableTMAs = findAllowableTMAs(obj, idxInitiation)
            allowableTMAs = [];
            for idxTMA = 1:obj.numTMAs
                if ~isnan(obj.TMAs(idxTMA).tau(idxInitiation))
                    allowableTMAs = [allowableTMAs; idxTMA];
                end
            end
        end
        
        function numLeaves = drawTree(obj, rootNodes, rootNodesTMAIdxs, nodeMap, nodeMapIdxs, maxDepth, numLeaves)
            labelTMAs = false;
            
            [numLeaves,nodeMap, nodeMapIdxs] = createDrawnTree(obj, rootNodes, rootNodesTMAIdxs, nodeMap, nodeMapIdxs, maxDepth, numLeaves);
            figure('units','normalized','outerposition',[0 0 1 1])
            
            treeplot(nodeMap,'ob')
            
            if (labelTMAs)
                %count = size(nodeMap,2);
                [x,y] = treelayout(nodeMap);
                x = x';
                y = y';
                name1 = cellstr(num2str(nodeMapIdxs'));
                text(x(:,1), y(:,1), name1, 'VerticalAlignment','bottom','HorizontalAlignment','right');
                title({'Level Lines'},'FontSize',12,'FontName','Times New Roman');
            end
        end
        
        function [numLeaves, nodeMap, nodeMapIdxs] = createDrawnTree(dom, rootNodes, rootNodesTMAIdxs, nodeMap, nodeMapIdxs, maxDepth, numLeaves)
            if (numLeaves == -1)
                numLeaves = 0;
            end
            
            if (maxDepth > 0)
                numNodes = length(nodeMap);
                
                if (numNodes == 0)
                    nodeMap = [0];
                    nodeMapIdxs = rootNodes; %TMA index associated with this tree node
                end
                
                idxTemp = 1;
                for rootNode = rootNodes
                    rootNodeTMAIdx = rootNodesTMAIdxs(idxTemp);
                    %get all children of the root node
                    rootNodesTMAIdxsNext = dom.TMAs(rootNodeTMAIdx).allowableChildTMAIdxs; %TMA idx
                    numChildren = length(rootNodesTMAIdxsNext);
                    rootNodesNext = (1:numChildren) + length(nodeMap);%tree idx (for visualization only)
                    
                    nodeMap = [nodeMap ones(1,numChildren).*rootNode];
                    nodeMapIdxs = [nodeMapIdxs rootNodesTMAIdxsNext'];
                    
                    [numLeaves,nodeMap, nodeMapIdxs] = createDrawnTree(dom, rootNodesNext, rootNodesTMAIdxsNext, nodeMap, nodeMapIdxs, maxDepth - 1, numLeaves);
                    idxTemp = idxTemp + 1;
                end
                
                %at leaf-level. Report the number of leaves.
                if (maxDepth == 1)
                    idxTemp = 1;
                    for rootNode = rootNodes
                        rootNodeTMAIdx = rootNodesTMAIdxs(idxTemp);
                        %get all children of the root node
                        rootNodesTMAIdxsNext = dom.TMAs(rootNodeTMAIdx).allowableChildTMAIdxs; %TMA idx
                        numLeaves = numLeaves + length(rootNodesTMAIdxsNext);
                        idxTemp = idxTemp + 1;
                    end
                end
            end
        end
        
    end
end