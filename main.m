function main()
    clc
    close all
    
    dom = Domain();
    
    %policy eval tests
    table = ones(13,13);
    myPolicy = Policy(table);
    dom.setPolicyForAllAgents(myPolicy);
    
%     constructTree(dom);
%     newRootNodesTMAIdxs = 1; %assume TMA1 conducted at root
%     numLeaves = [];
%     maxDepths = 1;
%     for maxDepth = maxDepths
%         numLeaves = [numLeaves drawTree(dom, 1, newRootNodesTMAIdxs, [], [], maxDepth, -1)];
%     end
% %     set(gcf,'PaperPositionMode','auto')
% %     print('-dpng','-zbuffer','-r200','searchtree_1agent.png')
% 
%     figure
%     plot(maxDepths, numLeaves);
%     xlabel('Search depth');
%     ylabel('# of branches');
%     grid on
% %     set(gcf,'PaperPositionMode','auto')
% %     print('-dpng','-zbuffer','-r200','branches_vs_depth.png')
%     %     printAllowableTMAs(dom);
%     close all

%     dom.TMAs(1)
%     dom.TMAs(1).sampleTau(dom.beliefNodes(2))
%     dom.beliefNodes(1).xE
end

function printAllowableTMAs(dom)
    for idxTMA = 1:dom.numTMAs
        idxTMA
        dom.TMAs(idxTMA).allowableChildTMAIdxs
    end
end

function constructTree(dom)
    for idxTMA = 1:dom.numTMAs
        %if termination set contains NaN, then agent's termination set is equal to its initiation set
        if isnan(dom.TMAs(idxTMA).bTerm)
            %find all initiation set elements that are allowable (or, are not equal to NaN)
            allTerminationsIdx = find(isnan(dom.TMAs(idxTMA).tau)==0);
            for idxTermination = allTerminationsIdx
                dom.TMAs(idxTMA).allowableChildTMAIdxs = findAllowableTMAs(dom, idxTermination);
            end
        else
            %there is a specific termination set associated with the TMA
            for idxTermination = 1:dom.numBeliefNodes
                if (dom.TMAs(idxTMA).bTerm(idxTermination)==1)
                    dom.TMAs(idxTMA).allowableChildTMAIdxs = findAllowableTMAs(dom, idxTermination);
                end
            end
        end
    end
end

function allowableTMAs = findAllowableTMAs(dom, idxInitiation)
    allowableTMAs = [];
    for idxTMA = 1:dom.numTMAs
        if ~isnan(dom.TMAs(idxTMA).tau(idxInitiation))
            allowableTMAs = [allowableTMAs; idxTMA];
        end
    end
end

function numLeaves = drawTree(dom, rootNodes, rootNodesTMAIdxs, nodeMap, nodeMapIdxs, maxDepth, numLeaves)
    labelTMAs = false;
    
    [numLeaves,nodeMap, nodeMapIdxs] = createDrawnTree(dom, rootNodes, rootNodesTMAIdxs, nodeMap, nodeMapIdxs, maxDepth, numLeaves);
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