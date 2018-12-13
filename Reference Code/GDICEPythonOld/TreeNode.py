from __future__ import absolute_import, print_function, division


"""
Class to represent a node in a policy tree
Instance Variables:
  parent: Parent to this node in tree
  children: Direct children of this node in tree
"""
class TreeNode(object):

    """
    Constructor
    """
    def __init__(self):
        self.parent = None
        self.children = []

    """
    Add a child to this node
    Input:
      child: TreeNode object that is this node's child
    """
    def addChild(self, child):
        if isinstance(child, TreeNode):
            self.children.append(child)
            self.children[-1].setParent(self)
        else:
            raise ValueError("Can't add non TreeNode object as child")

    """
    Set this node's parent
    Input:
      parent: parent TreeNode
    """
    def setParent(self, parent):
        self.parent = parent