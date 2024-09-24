
import time
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

def calculate_similarity(person1, person2):
    #calculate the similarity score based on the inverse of the sum of absolute differences in ratings
    score = sum(abs(x - y) for x, y in zip(person1, person2))
    return 1 / (1 + score)  # Using inverse to make lower differences yield higher scores

def count_common_interests(person1, person2):
    #count how many interests have the same rating for both people
    return sum(1 for x, y in zip(person1, person2) if x == y)

def find_best_pairs(num_people, ratings):
    #find the best pairing based on maximizing overall similarity
    people = list(range(num_people))
    best_score = 0
    best_pairs = []

    # Check all permutations of people for the highest scoring configuration
    for perm in permutations(people):
        pairs = [(perm[i], perm[i + 1]) for i in range(0, len(perm), 2)]
        score = sum(calculate_similarity(ratings[p1], ratings[p2]) for p1, p2 in pairs)
        if score > best_score:
            best_score = score
            best_pairs = pairs

    return best_pairs, score

def visualize_relationships(num_people, ratings, best_pairs):
    #visualize the relationships based on common interests ratings
    G = nx.Graph()
    # Add nodes
    for i in range(num_people):
        G.add_node(i, label=f"Person {i}")

    # Add edges for common interests
    for i in range(num_people):
        for j in range(i + 1, num_people):
            common_interests = count_common_interests(ratings[i], ratings[j])
            if common_interests > 0:
                G.add_edge(i, j, weight=common_interests, color='blue')

    #highlight best pairs with red edges
    for p1, p2 in best_pairs:
        G.add_edge(p1, p2, weight=count_common_interests(ratings[p1], ratings[p2]), color='red')

    pos = nx.spring_layout(G)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    labels = nx.get_edge_attributes(G, 'weight')
    
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v}" for k, v in labels.items()})
    
    plt.title("Graph showing number of common ratings per interest")
    plt.show()


def main():
    # record start time
    start = time.time()

    interests = ["Cleanliness", "Noise Sensitivity", "Hospitality", "Preferred Work Time", "Social Orientation", "Smoking Habits"]

    # First 8 men participants
    ratings = [[3,3,2,2,3,5],[3,4,2,3,3,5],[2,3,2,3,4,4],[2,3,1,2,3,4],
            [3,3,2,2,1,5],[2,5,2,2,2,5],[2,4,2,2,2,3],[2,4,2,3,3,5]]

    num_people = len(ratings)

    # Display ratings
    print("Ratings per person:")
    for idx, rate in enumerate(ratings):
        print(f"Person {idx}: {dict(zip(interests, rate))}")

    # Find best pairs based on interest ratings
    best_pairs, score = find_best_pairs(num_people, ratings)
    print("\nBest pairs based on interest similarity:")
    for p1, p2 in best_pairs:
        print(f"Pair ({p1}, {p2}) with similarity score {calculate_similarity(ratings[p1], ratings[p2]):.2f}")
    print(f"Total Similarity Score: {score:.2f}")

    #show relationships
    visualize_relationships(num_people, ratings, best_pairs)

    end = time.time()
    print("The execution time is :", (end - start) * 1000, "ms")

main()