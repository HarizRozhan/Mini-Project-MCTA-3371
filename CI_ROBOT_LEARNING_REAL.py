import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from collections import deque

# ==========================================
# 1. CONFIGURATION
# ==========================================
LAYOUTS = {"Simple": (9, 9), "Intermediate": (13, 13), "Complex": (19, 19)}
SPEEDS = {"Simple": 0.1, "Intermediate": 0.05, "Complex": 0.02}

# Distinct Color Palette
COLOR_WALL = '#2C3E50'
COLOR_PATH = '#ECF0F1'
COLOR_START = '#F1C40F'  # Bright Yellow
COLOR_GOAL = '#8E44AD'  # Vibrant Purple
COLOR_ROBOT_FUZZY = '#E74C3C'  # Red
COLOR_ROBOT_GA = '#3498DB'  # Blue
COLOR_EXPLORED = '#AED6F1'  # Blue Trail
COLOR_BACKTRACK = '#BDC3C7'  # Grey Dead-ends
COLOR_GA_PATH = '#27AE60'  # Green Optimized Dots


# ==========================================
# 2. HELPER FUNCTIONS (Python 3.7 Compatible)
# ==========================================
def get_continuous_path(waypoints, maze):
    """Interpolates waypoints into a tile-by-tile path."""
    full_path = []
    rows, cols = len(maze), len(maze[0])
    for i in range(len(waypoints) - 1):
        start_pt, end_pt = waypoints[i], waypoints[i + 1]
        if start_pt == end_pt: continue
        queue = deque([start_pt])
        came_from = {start_pt: None}
        found = False
        while queue:
            curr = queue.popleft()
            if curr == end_pt:
                found = True
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    if maze[ny][nx] == 0 and (nx, ny) not in came_from:
                        came_from[(nx, ny)] = curr
                        queue.append((nx, ny))
        if found:
            segment = []
            curr = end_pt
            while curr is not None:
                segment.append(curr)
                curr = came_from.get(curr)
            full_path.extend(segment[::-1][:-1])
    full_path.append(waypoints[-1])
    return full_path


# ==========================================
# 3. SOFT COMPUTING: FUZZY & ELITE GA
# ==========================================
def fuzzy_decision(neighbors, goal):
    """Fuzzy heuristic picking neighbor closest to goal."""
    scored = []
    for n in neighbors:
        dist = abs(n[0] - goal[0]) + abs(n[1] - goal[1])
        score = 1.0 / (dist + 1)
        scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def genetic_optimization(start, goal, timeline, maze, layout_type):
    """Elite GA: Chronologically optimizes the discovery path."""
    gens = 120 if layout_type == "Complex" else 60
    pop_size = 40
    # Initial Population
    population = []
    for _ in range(pop_size):
        pts = random.sample(timeline, min(len(timeline), 6))
        pts.sort(key=lambda x: timeline.index(x))  # Force Chronology
        population.append([start] + pts + [goal])

    for _ in range(gens):
        # Fitness based on final step count
        population.sort(key=lambda p: len(get_continuous_path(p, maze)))
        next_gen = population[:20]
        while len(next_gen) < pop_size:
            parent = random.choice(next_gen)
            child = list(parent)
            if len(child) > 3:
                idx = random.randint(1, len(child) - 2)
                prev_t_idx = timeline.index(child[idx - 1])
                if prev_t_idx < len(timeline) - 1:
                    child[idx] = random.choice(timeline[prev_t_idx:])
            mid = list(set(child[1:-1]))
            mid.sort(key=lambda x: timeline.index(x))
            next_gen.append([start] + mid + [goal])
        population = next_gen
    return population[0]


# ==========================================
# 4. MAZE GENERATION
# ==========================================
def generate_maze(rows, cols):
    maze = [[1 for _ in range(cols)] for _ in range(rows)]
    stack = [(1, 1)];
    maze[1][1] = 0
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < cols - 1 and 1 <= ny < rows - 1 and maze[ny][nx] == 1:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            maze[cy + dy // 2][cx + dx // 2] = 0;
            maze[ny][nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    return maze


# ==========================================
# 5. MAIN SIMULATION ENGINE
# ==========================================
def run_comparison(layout_name, rows, cols):
    maze = generate_maze(rows, cols)
    start, goal = (1, 1), (cols - 2, rows - 2)
    speed = SPEEDS[layout_name]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title(f"Elite Comparison: {layout_name}")
    fig.suptitle(f"Fuzzy exploration vs Fuzzy-GA Optimization ({layout_name})", fontsize=16)

    # Step Counters
    fuzzy_txt = ax1.text(cols / 2, rows + 0.5, "Steps: 0", ha='center', color='red', weight='bold')
    ga_txt = ax2.text(cols / 2, rows + 0.5, "Steps: 0", ha='center', color='green', weight='bold')

    for ax, title in zip([ax1, ax2], ["Trial 1: Fuzzy logic", "Trial 2: Fuzzy-GA Optimized"]):
        ax.set_title(title)
        ax.set_xlim(0, cols);
        ax.set_ylim(0, rows);
        ax.invert_yaxis();
        ax.axis('off')
        for y in range(rows):
            for x in range(cols):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=COLOR_WALL if maze[y][x] == 1 else COLOR_PATH))
        ax.add_patch(patches.Rectangle(start, 1, 1, facecolor=COLOR_START, zorder=5))
        ax.add_patch(patches.Rectangle(goal, 1, 1, facecolor=COLOR_GOAL, zorder=5))

    robot1 = patches.Circle((start[0] + 0.5, start[1] + 0.5), 0.35, facecolor=COLOR_ROBOT_FUZZY, zorder=10,
                            edgecolor='black')
    robot2 = patches.Circle((start[0] + 0.5, start[1] + 0.5), 0.35, facecolor=COLOR_ROBOT_GA, zorder=10,
                            edgecolor='black')
    ax1.add_patch(robot1);
    ax2.add_patch(robot2)

    # --- TRIAL 1: FUZZY DISCOVERY ---
    stack, visited, timeline = [start], {start}, [start]
    steps_fuzzy = 0
    plt.ion();
    plt.show()

    while stack:
        curr = stack[-1];
        steps_fuzzy += 1
        fuzzy_txt.set_text(f"Exploration Steps: {steps_fuzzy}")
        if curr == goal: break

        # Check neighbors
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0 and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if neighbors:
            # 20% Chance of 'Discovery Error' (Picking random over best neighbor)
            next_c = random.choice(neighbors) if random.random() < 0.2 else fuzzy_decision(neighbors, goal)
            visited.add(next_c);
            timeline.append(next_c);
            stack.append(next_c)
            ax1.add_patch(patches.Rectangle(next_c, 1, 1, facecolor=COLOR_EXPLORED, alpha=0.5, zorder=1))
            robot1.center = (next_c[0] + 0.5, next_c[1] + 0.5)
        else:
            ax1.add_patch(patches.Rectangle(curr, 1, 1, facecolor=COLOR_BACKTRACK, alpha=0.8, zorder=1))
            stack.pop()
            if stack: robot1.center = (stack[-1][0] + 0.5, stack[-1][1] + 0.5)
        plt.pause(speed)

    # --- TRIAL 2: ELITE GA ---
    print(f"[{layout_name}] Optimizing...")
    best_wp = genetic_optimization(start, goal, timeline, maze, layout_name)
    smooth_path = get_continuous_path(best_wp, maze)

    steps_ga = 0
    for cell in smooth_path:
        steps_ga += 1
        ga_txt.set_text(f"Optimized Steps: {steps_ga}")
        if cell != start and cell != goal:
            ax2.add_patch(patches.Circle((cell[0] + 0.5, cell[1] + 0.5), 0.15, color=COLOR_GA_PATH, zorder=4))
        robot2.center = (cell[0] + 0.5, cell[1] + 0.5)
        plt.pause(speed)

    print(f"[{layout_name}] Efficiency Gain: {((steps_fuzzy - steps_ga) / steps_fuzzy) * 100:.1f}%")
    plt.ioff();
    plt.show()


if __name__ == "__main__":
    for name, size in LAYOUTS.items():
        run_comparison(name, size[0], size[1])