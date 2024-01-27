import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

class Node:
    def __init__(self, x, y, theta, cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = cost
        self.parent = parent

def hybrid_a_star(start, goal, obstacles, max_steering_angle, delta_t, v_max):
    open_set = [Node(start.x, start.y, start.theta, 0)]
    closed_set = set()

    while open_set:
        current_node = min(open_set, key=lambda node: node.cost)
        open_set.remove(current_node)

        if (current_node.x, current_node.y) == (goal.x, goal.y):
            return reconstruct_path(current_node)

        for delta_theta in np.arange(-max_steering_angle, max_steering_angle + 0.1, max_steering_angle / 2):
            next_theta = current_node.theta + delta_theta
            next_x = current_node.x + v_max * delta_t * cos(next_theta)
            next_y = current_node.y + v_max * delta_t * sin(next_theta)

            new_node = Node(next_x, next_y, next_theta, current_node.cost + 1, current_node)

            if (new_node.x, new_node.y) not in obstacles and new_node not in closed_set:
                open_set.append(new_node)

        closed_set.add((current_node.x, current_node.y))

    return None  # No path found

def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def simulate_path(path):
    plt.figure(figsize=(8, 8))
    plt.plot([point[0] for point in path], [point[1] for point in path], marker='o')
    plt.title('Hybrid A* Path Planning')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

if __name__ == "__main__":
    start = Node(0, 0, pi/4, 0)
    goal = Node(5, 5, 3*pi/4, 0)
    obstacles = {(2, 2), (3, 3), (4, 4)}
    max_steering_angle = pi/4
    delta_t = 0.1
    v_max = 1.0

    path = hybrid_a_star(start, goal, obstacles, max_steering_angle, delta_t, v_max)

    if path:
        print("Path found:", path)
        simulate_path(path)
    else:
        print("No path found.")
