import networkx as nx
from simulation import create_graph, get_influencers_cost

chic_choc_path = 'chic_choc_data.csv'
cost_path = 'costs.csv'
G = create_graph(chic_choc_path)

#FIRST PART CODE

closeness = nx.closeness_centrality(G)
degree = dict(G.degree())

# Give a grade from 0 to 1000 for each node based on their price and degree
closeness_scaled = {
    node: (closeness[node] - min(closeness.values())) / (max(closeness.values()) - min(closeness.values())) * 1000 for
    node in closeness}
degree_scaled = {node: (degree[node] - min(degree.values())) / (max(degree.values()) - min(degree.values())) * 1000 for
                 node in degree}

# Get the results into list
result = {}
for node in G.nodes():
    node_list = []
    node_list.append(node)
    res = ((degree_scaled[node] + closeness_scaled[node]) / 2) - get_influencers_cost(cost_path, node_list)
    result[node] = res

# Sorted the nodes by most influencers one.
top_20 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:100]

print("Top 20 most influencers nodes :")
for i, (node, score) in enumerate(top_20):
    print(f"{i + 1}. Node {node} - Score : {score:.2f}")


#SECOND PART CODE


def find_best_node(G, S):
    best_node = None
    best_value = -float('inf')
    total_neigh = set()
    for i in S:
        total_neigh.update(list(G.neighbors(i)))
    for node in G.nodes:
        if node not in S:
            value = set(G.neighbors(node))
            intersection = value & total_neigh
            node_list = []
            node_list.append(node)
            gap = len(value) - len(intersection)
            node_list.remove(node)
            if gap > best_value:
                best_node = node
                best_value = gap
    return best_node


def set_cover(G, k):
    S = [348]
    for i in range(k):
        si = find_best_node(G, S)
        S.append(si)
    return S

