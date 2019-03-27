from graphviz import Digraph

# Convert a discrete FSC policy to a visualized graph
def convertFSCToGraph(fsc, name='Policy'):
    dot = Digraph(comment=name)
    [dot.node(str(i), 'A'+str(fsc.actionTransitions[i])) for i in range(fsc.numNodes)]  # Add nodes
    [dot.edge(str(i), str(fsc.nodeObservationTransitions[j, i]), 'O'+str(j)) for i in range(fsc.numNodes) for j in range(fsc.numObservations)] # Add edges
    try:
        dot.render(name+'.gv')
    except:
        dot.save(name+'.gv')

    return dot