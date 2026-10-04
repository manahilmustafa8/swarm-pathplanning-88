import random
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. STUDENT SEED
# ============================================================

SEED = 88
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. GENERATE UNIQUE PROBLEM INSTANCE
# ============================================================

def generate_problem(seed):
    rng = random.Random(seed)

    # Grid size is also generated from the student's seed
    grid_size = rng.randint(18, 22)

    obstacle_rate = 0.22

    while True:

        # Generate obstacles randomly
        obstacles = set()

        for row in range(grid_size):
            for col in range(grid_size):
                if rng.random() < obstacle_rate:
                    obstacles.add((row, col))

        # Generate start and goal from free cells
        free_cells = [
            (row, col)
            for row in range(grid_size)
            for col in range(grid_size)
            if (row, col) not in obstacles
        ]

        start = rng.choice(free_cells)
        goal = rng.choice(free_cells)

        if start == goal:
            continue

        # Make sure a path is actually possible
        if bfs_exists(start, goal, obstacles, grid_size):
            return grid_size, obstacles, start, goal


# ============================================================
# 3. CHECK WHETHER A PATH EXISTS
# ============================================================

def bfs_exists(start, goal, obstacles, grid_size):

    queue = deque([start])
    visited = {start}

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1)
    ]

    while queue:

        current = queue.popleft()

        if current == goal:
            return True

        for dr, dc in directions:

            nr = current[0] + dr
            nc = current[1] + dc

            neighbor = (nr, nc)

            if (
                0 <= nr < grid_size
                and 0 <= nc < grid_size
                and neighbor not in obstacles
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)

    return False


# ============================================================
# 4. BRESENHAM LINE
#    Converts PSO waypoint positions into grid cells
# ============================================================

def bresenham(start, end):

    r0, c0 = start
    r1, c1 = end

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    error = dr - dc

    cells = []

    r, c = r0, c0

    while True:

        cells.append((r, c))

        if (r, c) == (r1, c1):
            break

        e2 = 2 * error

        if e2 > -dc:
            error -= dc
            r += sr

        if e2 < dr:
            error += dr
            c += sc

    return cells


# ============================================================
# 5. DECODE PARTICLE INTO A PATH
# ============================================================

def decode_particle(position, start, goal, grid_size, num_waypoints):

    points = [start]

    for i in range(num_waypoints):

        row = int(round(position[2 * i]))
        col = int(round(position[2 * i + 1]))

        row = max(0, min(grid_size - 1, row))
        col = max(0, min(grid_size - 1, col))

        points.append((row, col))

    points.append(goal)

    path = []

    for point_a, point_b in zip(points, points[1:]):

        segment = bresenham(point_a, point_b)

        if path:
            path.extend(segment[1:])
        else:
            path.extend(segment)

    return path


# ============================================================
# 6. FITNESS FUNCTION
# ============================================================

def fitness(
    position,
    start,
    goal,
    obstacles,
    grid_size,
    num_waypoints
):

    path = decode_particle(
        position,
        start,
        goal,
        grid_size,
        num_waypoints
    )

    collision_count = sum(
        cell in obstacles
        for cell in path
    )

    # Euclidean path length
    path_length = 0

    for i in range(len(path) - 1):

        path_length += math.dist(
            path[i],
            path[i + 1]
        )

    # Very large penalty for hitting obstacles
    collision_penalty = 1000 * collision_count

    total_cost = path_length + collision_penalty

    return total_cost, path_length, collision_count, path


# ============================================================
# 7. PATH SIMPLIFICATION
# ============================================================

def simplify_path(path, obstacles):

    simplified = [path[0]]

    current_index = 0

    while current_index < len(path) - 1:

        best_index = current_index + 1

        for j in range(
            current_index + 1,
            len(path)
        ):

            segment = bresenham(
                path[current_index],
                path[j]
            )

            collision = any(
                cell in obstacles
                for cell in segment
            )

            if not collision:
                best_index = j

        simplified.append(path[best_index])

        current_index = best_index

    # Expand simplified path back into grid cells
    final_path = []

    for a, b in zip(
        simplified,
        simplified[1:]
    ):

        segment = bresenham(a, b)

        if final_path:
            final_path.extend(segment[1:])
        else:
            final_path.extend(segment)

    return final_path


# ============================================================
# 8. PARTICLE SWARM OPTIMIZATION
# ============================================================

def pso(
    start,
    goal,
    obstacles,
    grid_size
):

    # Number of intermediate waypoints
    num_waypoints = 25

    dimensions = num_waypoints * 2

    # PSO parameters
    num_particles = 100
    max_iterations = 500

    inertia_max = 0.9
    inertia_min = 0.4

    c1 = 1.7
    c2 = 1.7

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------
    # Initial waypoint positions along approximate straight line
    # --------------------------------------------------------

    base_position = []

    for i in range(num_waypoints):

        t = (i + 1) / (num_waypoints + 1)

        row = start[0] + t * (goal[0] - start[0])
        col = start[1] + t * (goal[1] - start[1])

        base_position.extend([row, col])

    base_position = np.array(base_position)

    # --------------------------------------------------------
    # Initialize particles
    # --------------------------------------------------------

    particles = np.tile(
        base_position,
        (num_particles, 1)
    )

    particles += rng.normal(
        0,
        4,
        particles.shape
    )

    particles = np.clip(
        particles,
        0,
        grid_size - 1
    )

    # Initial velocities
    velocities = rng.uniform(
        -1.5,
        1.5,
        (num_particles, dimensions)
    )

    # --------------------------------------------------------
    # Personal best
    # --------------------------------------------------------

    personal_best = particles.copy()

    personal_cost = np.array([
        fitness(
            particle,
            start,
            goal,
            obstacles,
            grid_size,
            num_waypoints
        )[0]
        for particle in particles
    ])

    # --------------------------------------------------------
    # Global best
    # --------------------------------------------------------

    best_index = np.argmin(personal_cost)

    global_best = personal_best[best_index].copy()
    global_cost = personal_cost[best_index]

    # --------------------------------------------------------
    # PSO iterations
    # --------------------------------------------------------

    for iteration in range(max_iterations):

        # Decreasing inertia weight
        inertia = (
            inertia_max
            - (inertia_max - inertia_min)
            * iteration
            / (max_iterations - 1)
        )

        r1 = rng.random(
            (num_particles, dimensions)
        )

        r2 = rng.random(
            (num_particles, dimensions)
        )

        # PSO velocity update
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best - particles)
            + c2 * r2 * (global_best - particles)
        )

        velocities = np.clip(
            velocities,
            -3,
            3
        )

        # PSO position update
        particles += velocities

        particles = np.clip(
            particles,
            0,
            grid_size - 1
        )

        # Evaluate particles
        current_cost = np.array([
            fitness(
                particle,
                start,
                goal,
                obstacles,
                grid_size,
                num_waypoints
            )[0]
            for particle in particles
        ])

        # Update personal best
        improved = current_cost < personal_cost

        personal_best[improved] = particles[improved]
        personal_cost[improved] = current_cost[improved]

        # Update global best
        best_index = np.argmin(current_cost)

        if current_cost[best_index] < global_cost:

            global_cost = current_cost[best_index]

            global_best = particles[
                best_index
            ].copy()

        if (iteration + 1) % 50 == 0:

            print(
                f"Iteration {iteration + 1:3d} | "
                f"Best Cost = {global_cost:.3f}"
            )

    # Get final PSO path
    _, raw_length, collisions, raw_path = fitness(
        global_best,
        start,
        goal,
        obstacles,
        grid_size,
        num_waypoints
    )

    # Simplify the PSO path
    final_path = simplify_path(
        raw_path,
        obstacles
    )

    final_length = sum(
        math.dist(
            final_path[i],
            final_path[i + 1]
        )
        for i in range(len(final_path) - 1)
    )

    return final_path, final_length, collisions


# ============================================================
# 9. VISUALIZATION
# ============================================================

def visualize(
    grid_size,
    obstacles,
    start,
    goal,
    path
):

    plt.figure(figsize=(9, 9))

    # Draw grid
    for i in range(grid_size + 1):

        plt.axhline(
            i - 0.5,
            linewidth=0.5,
            alpha=0.3
        )

        plt.axvline(
            i - 0.5,
            linewidth=0.5,
            alpha=0.3
        )

    # Draw obstacles
    for row, col in obstacles:

        plt.fill(
            [
                col - 0.5,
                col + 0.5,
                col + 0.5,
                col - 0.5
            ],
            [
                row - 0.5,
                row - 0.5,
                row + 0.5,
                row + 0.5
            ],
            color="black"
        )

    # Draw path
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]

    plt.plot(
        path_cols,
        path_rows,
        linewidth=2,
        marker="o",
        markersize=3,
        label="PSO Path"
    )

    # Start
    plt.scatter(
        start[1],
        start[0],
        s=180,
        marker="o",
        label="Start"
    )

    # Goal
    plt.scatter(
        goal[1],
        goal[0],
        s=180,
        marker="*",
        label="Goal"
    )

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(grid_size - 0.5, -0.5)

    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.title(
        f"PSO 2D Grid Path Planning\n"
        f"Seed = {SEED}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "output/best_path.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("PSO 2D GRID PATH PLANNING")
    print("=" * 55)

    print(f"Random Seed: {SEED}")

    # Generate unique problem
    grid_size, obstacles, start, goal = generate_problem(SEED)

    print(f"Grid Size: {grid_size} x {grid_size}")
    print(f"Number of Obstacles: {len(obstacles)}")
    print(f"Start Point: {start}")
    print(f"Goal Point: {goal}")

    print("\nRunning PSO...\n")

    # Run PSO
    path, path_length, collisions = pso(
        start,
        goal,
        obstacles,
        grid_size
    )

    print("\n" + "=" * 55)
    print("FINAL RESULT")
    print("=" * 55)

    print(f"Path Length: {path_length:.3f}")
    print(f"Obstacle Collisions: {collisions}")
    print(f"Number of Path Cells: {len(path)}")

    print("\nBest Path:")

    for cell in path:
        print(cell)

    # Visualization
    visualize(
        grid_size,
        obstacles,
        start,
        goal,
        path
    )