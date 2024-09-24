
#import time
#import numpy as np
#import networkx as nx
# import matplotlib.pyplot as plt

def count_common_interests(person1, person2, ratings):
    #calculate the similarity score based on the inverse of the sum of absolute differences in ratings
    return sum(1 for x, y in zip(ratings[person1], ratings[person2]) if x == y)

# def visualize_relationships(num_people, ratings, pairs):
#     G = nx.Graph()
#     # Add all nodes
#     for i in range(num_people):
#         G.add_node(i, label=f"Person {i}")

#     # Add edges for all possible pairs with common interests count as the weight
#     for i in range(num_people):
#         for j in range(i + 1, num_people):
#             common_interests = count_common_interests(i, j, ratings)
#             if common_interests > 0:
#                 G.add_edge(i, j, weight=common_interests, color='black')  # Default color

#     # Update edges for optimal pairs to be red without changing the weight
#     for p1, p2 in pairs:
#         if G.has_edge(p1, p2):
#             # Update the color of the edge if it exists
#             G[p1][p2]['color'] = 'red'
#         else:
#             # Add the edge if it does not exist
#             common_interests = count_common_interests(p1, p2, ratings)
#             G.add_edge(p1, p2, weight=common_interests, color='red')

#     pos = nx.spring_layout(G)  # positions for all nodes
#     edge_colors = [G[u][v]['color'] for u, v in G.edges()]

#     # Draw the graph
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=edge_colors, width=1)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v}" for k, v in edge_labels.items()})

#     plt.title("Graph showing Pairings with Common Interests")
#     plt.show()

def calculate_similarity(person1, person2, ratings):
    """ Calculate similarity score based on the inverse of the sum of absolute differences in ratings. """
    score = sum(abs(x - y) for x, y in zip(ratings[person1], ratings[person2]))
    return 1 / (1 + score)  # Using inverse to make lower differences yield higher scores


def recursive_pairing(people, ratings, current_pairs=[]):
    if not people:
        score = sum(calculate_similarity(p1, p2, ratings) for p1, p2 in current_pairs)
        return score, current_pairs

    best_score = float('-inf')
    best_configuration = None
    first = people[0]

    for second in people[1:]:
        new_pairs = current_pairs + [(first, second)]
        new_people = [p for p in people if p not in (first, second)]
        score, configuration = recursive_pairing(new_people, ratings, new_pairs)
        
        if score > best_score:
            best_score = score
            best_configuration = configuration

    return best_score, best_configuration

def main():
    # Measure execution time
    #start = time.time()

    #12 male# particpants
    ratings = [[3,3,2,2,3,5],[3,4,2,3,3,5],[2,3,2,3,4,4],[2,3,1,2,3,4],
            [3,3,2,2,1,5],[2,5,2,2,2,5],[2,4,2,2,2,3],[2,4,2,3,3,5],
            [1,1,3,2,4,3],[1,4,1,5,5,2],[5,3,3,4,3,5],[4,3,3,4,4,3]]

    num_people = len(ratings)

    people = list(range(num_people))
    score, pairs = recursive_pairing(people, ratings)
    print("The best pairs are:")
    for p1, p2 in pairs:
        sim = calculate_similarity(p1, p2, ratings)
        print(f"Pair ({p1}, {p2}) with similarity score {sim:.2f}")
    print("Total similarity score:", score)

    #end = time.time()
    #visualize_relationships(num_people, ratings, pairs)

    #print("The execution time is:", (end-start) * 1000, "ms")

main()