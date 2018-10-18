classdef TreeNode < handle
    properties
        parent = [];
        children = [];
    end
    methods
        function obj = TreeNode()
        end
        
        function addChild(obj, child)
            if (isa(child,'TreeNode'))
                obj.children = [obj.children; child];
                child.setParent(obj);
            else
                sprintf('Error! Tried to add a non "TreeNode" object as a child!')
            end
        end
        
        function setParent(obj, parent)
            obj.parent = parent;
        end
    end
end