
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

def calculate_similarity(person1, person2):
    # Here calculate the similarity score based on the inverse of the sum of absolute differences in ratings. """
    return 1 / (1 + sum(abs(x - y) for x, y in zip(person1, person2)))
    # return np.sqrt(np.sum((np.array(person1) - np.array(person2)) ** 2))

def count_common_interests(person1, person2):
    #Count how many interests have the same rating for both people
    return sum(1 for x, y in zip(person1, person2) if x == y)

def greedy_pairing(people):
    num_people = len(people)
    matched = set()
    pairs = []
    similarities = {}

    # Calculate all pairwise similarities for matching
    for i, j in combinations(range(num_people), 2):
        similarity = calculate_similarity(people[i], people[j])
        similarities[(i, j)] = similarity

    # Sort pairs by descending similarity for greedy matching
    sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Select pairs greedily
    for (i, j), sim in sorted_pairs:
        if i not in matched and j not in matched:
            pairs.append((i, j))
            matched.update([i, j])
            if len(matched) >= num_people:
                break

    return pairs

def visualize_relationships(pairs, people):
    #visualize the relationships based on common interests ratings
    G = nx.Graph()
    num_people = len(people)

    # Add nodes and edges with the count of common interests
    for i in range(num_people):
        for j in range(i + 1, num_people):
            common_interests = count_common_interests(people[i], people[j])
            if common_interests > 0:
                G.add_edge(i, j, weight=common_interests)

    # Highlight selected pairs with different color and edge style
    selected_edges = [(i, j) for i, j in pairs]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v}" for k, v in edge_labels.items()})
    nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=2)

    plt.title("Network Graph Showing Pair Selections and Common Interests")
    plt.show()

def main():
    # Measure execution time
    start = time.time()

    # Women paricipants
    people =  [[5,5,3,2,4,5],[3,4,4,3,3,2],[4,3,2,4,5,5],[3,4,2,2,3,5]]

    pairs = greedy_pairing(people)
    print("Pairs based on highest similarities:", pairs)
    end = time.time()

    visualize_relationships(pairs, people)

    print("Execution time:", (end - start) * 1000, "ms")

main()
