from collections import deque
import heapq
from math import inf

import numpy as np
from manim import *


# 0 表示可通行，1 表示墙
MAZE = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
]

MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node != start and node in parent:
        path.append(node)
        node = parent[node]
    if node != start:
        return []
    path.append(start)
    path.reverse()
    return path


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_step(current, frontier, visited, decision, extra=None, found=False):
    data = {
        "current": current,
        "frontier": list(frontier),
        "visited": set(visited),
        "decision": decision,
        "neighbors": [],
        "found": found,
        "extra": extra or {},
    }
    return data


def search_with_trace(maze, algorithm):
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    if maze[start[0]][start[1]] == 1 or maze[goal[0]][goal[1]] == 1:
        return [], []

    trace = []

    if algorithm == "bfs":
        visited = [[False] * cols for _ in range(rows)]
        parent = {}
        queue = deque([start])
        visited[start[0]][start[1]] = True
        order = [start]

        while queue:
            current = queue.popleft()
            cx, cy = current
            step = make_step(
                current,
                queue,
                order,
                "take oldest queued node",
                extra={"structure": "queue", "rule": "FIFO"},
                found=current == goal,
            )
            if current == goal:
                trace.append(step)
                return trace, reconstruct_path(parent, start, goal)

            for dx, dy in MOVES:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                if not (0 <= nx < rows and 0 <= ny < cols):
                    step["neighbors"].append((neighbor, "out", "越界，直接跳过"))
                    continue
                if maze[nx][ny] == 1:
                    step["neighbors"].append((neighbor, "wall", "是墙，不能进入"))
                    continue
                if visited[nx][ny]:
                    step["neighbors"].append(
                        (neighbor, "seen", "已访问过，避免重复搜索")
                    )
                    continue
                visited[nx][ny] = True
                parent[neighbor] = current
                queue.append(neighbor)
                order.append(neighbor)
                step["neighbors"].append((neighbor, "push", "首次发现，加入队列尾部"))
            trace.append(step)
        return trace, []

    if algorithm == "dfs":
        visited = [[False] * cols for _ in range(rows)]
        parent = {}
        stack = [start]
        order = [start]
        active = set()

        while stack:
            current = stack.pop()
            if current in active:
                continue
            active.add(current)
            cx, cy = current
            step = make_step(
                current,
                stack,
                order,
                "take newest stacked node",
                extra={"structure": "stack", "rule": "LIFO"},
                found=current == goal,
            )
            if current == goal:
                trace.append(step)
                return trace, reconstruct_path(parent, start, goal)

            if visited[cx][cy]:
                continue
            visited[cx][cy] = True

            for dx, dy in reversed(MOVES):
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                if not (0 <= nx < rows and 0 <= ny < cols):
                    step["neighbors"].append((neighbor, "out", "越界，不能压栈"))
                    continue
                if maze[nx][ny] == 1:
                    step["neighbors"].append((neighbor, "wall", "是墙，不能压栈"))
                    continue
                if visited[nx][ny]:
                    step["neighbors"].append((neighbor, "seen", "已访问过"))
                    continue
                if neighbor not in parent:
                    parent[neighbor] = current
                stack.append(neighbor)
                order.append(neighbor)
                step["neighbors"].append((neighbor, "push", "压入栈顶，后续优先处理"))
            trace.append(step)
        return trace, []

    if algorithm == "dijkstra":
        dist = [[inf] * cols for _ in range(rows)]
        parent = {}
        heap = []
        order = [start]
        settled = [[False] * cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0
        heapq.heappush(heap, (0, start[0], start[1]))

        while heap:
            d, x, y = heapq.heappop(heap)
            current = (x, y)
            if settled[x][y]:
                continue
            settled[x][y] = True
            step = make_step(
                current,
                [(item[1], item[2]) for item in heap],
                order,
                "take node with minimum g",
                extra={"structure": "min-heap", "g": d},
                found=current == goal,
            )
            step["extra"]["dist"] = d
            if current == goal:
                trace.append(step)
                return trace, reconstruct_path(parent, start, goal)

            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if not (0 <= nx < rows and 0 <= ny < cols):
                    step["neighbors"].append((neighbor, "out", "越界"))
                    continue
                if maze[nx][ny] == 1:
                    step["neighbors"].append((neighbor, "wall", "是墙"))
                    continue
                nd = d + 1
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    parent[neighbor] = current
                    heapq.heappush(heap, (nd, nx, ny))
                    order.append(neighbor)
                    step["neighbors"].append(
                        (neighbor, "push", f"新距离 {nd} 更优，更新并入堆")
                    )
                else:
                    step["neighbors"].append(
                        (neighbor, "seen", f"已有更优距离 {dist[nx][ny]:g}")
                    )
            trace.append(step)
        return trace, []

    if algorithm == "astar":
        g_score = [[inf] * cols for _ in range(rows)]
        parent = {}
        heap = []
        order = [start]
        closed = [[False] * cols for _ in range(rows)]
        g_score[start[0]][start[1]] = 0
        heapq.heappush(heap, (heuristic(start, goal), 0, start[0], start[1]))

        while heap:
            f, g, x, y = heapq.heappop(heap)
            current = (x, y)
            if closed[x][y]:
                continue
            closed[x][y] = True
            step = make_step(
                current,
                [(item[2], item[3]) for item in heap],
                order,
                "take node with minimum f",
                extra={"structure": "min-heap", "g": g, "f": f, "h": f - g},
                found=current == goal,
            )
            if current == goal:
                trace.append(step)
                return trace, reconstruct_path(parent, start, goal)

            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if not (0 <= nx < rows and 0 <= ny < cols):
                    step["neighbors"].append((neighbor, "out", "越界"))
                    continue
                if maze[nx][ny] == 1:
                    step["neighbors"].append((neighbor, "wall", "是墙"))
                    continue
                tentative_g = g + 1
                if tentative_g < g_score[nx][ny]:
                    g_score[nx][ny] = tentative_g
                    parent[neighbor] = current
                    f2 = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(heap, (f2, tentative_g, nx, ny))
                    order.append(neighbor)
                    step["neighbors"].append(
                        (
                            neighbor,
                            "push",
                            f"g={tentative_g}, h={heuristic(neighbor, goal)}, f={f2}，因此更新",
                        )
                    )
                else:
                    step["neighbors"].append(
                        (neighbor, "seen", f"已有更优 g={g_score[nx][ny]:g}，不更新")
                    )
            trace.append(step)
        return trace, []

    raise ValueError(f"Unknown algorithm: {algorithm}")


class MazeSearchScene(Scene):
    algorithm = "bfs"
    ui_font = "SF Mono"

    def place_panel_content(self, box, content, left_pad=0.16, top_pad=0.18):
        content.next_to(box.get_top(), DOWN, buff=top_pad)
        content.align_to(box, LEFT)
        content.shift(RIGHT * left_pad)
        return content

    def make_cell(self, pos, size, fill_color, stroke_color=GREY_B, stroke_width=2):
        cell = Square(side_length=size)
        cell.set_fill(fill_color, opacity=1)
        cell.set_stroke(stroke_color, width=stroke_width)
        cell.move_to(pos)
        return cell

    def build_algorithm_panel(self, algorithm):
        panel = VGroup()
        if algorithm == "bfs":
            title = Text("BFS Panel", font=self.ui_font, font_size=16, weight=BOLD)
            fields = [
                ("Structure", "Queue / FIFO"),
                ("Rule", "Shortest steps first"),
            ]
        elif algorithm == "dfs":
            title = Text("DFS Panel", font=self.ui_font, font_size=16, weight=BOLD)
            fields = [
                ("Structure", "Stack / LIFO"),
                ("Rule", "Go deep first"),
            ]
        elif algorithm == "dijkstra":
            title = Text("Dijkstra Panel", font=self.ui_font, font_size=16, weight=BOLD)
            fields = [
                ("Structure", "Min-heap"),
                ("Rule", "Expand min g"),
            ]
        else:
            title = Text("A* Panel", font=self.ui_font, font_size=16, weight=BOLD)
            fields = [
                ("Structure", "Min-heap"),
                ("Rule", "Expand min f = g + h"),
            ]

        panel.add(title)
        for key, value in fields:
            row = VGroup(
                Text(f"{key}:", font=self.ui_font, font_size=12, color=GREY_A),
                Text(value, font=self.ui_font, font_size=12),
            ).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
            panel.add(row)
        panel.arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        frame = RoundedRectangle(
            width=3.55, height=1.85, corner_radius=0.12, color=GREY_B
        )
        frame.set_fill("#14181F", opacity=0.95)
        self.place_panel_content(frame, panel, left_pad=0.16, top_pad=0.18)
        frame.add(panel)
        return frame, panel

    def build_snapshot_panel(self, algorithm):
        if algorithm in {"bfs", "dfs"}:
            box = RoundedRectangle(
                width=3.55, height=1.85, corner_radius=0.12, color=GREY_B
            )
            title = Text(
                "Frontier Snapshot", font=self.ui_font, font_size=16, weight=BOLD
            )
            subtitle = Text(
                "Ordered as stored by the structure", font=self.ui_font, font_size=12
            )
            rows = VGroup(
                Text("Frontier: []", font=self.ui_font, font_size=12),
                Text("Action: waiting...", font=self.ui_font, font_size=12),
                Text("Next: N/A", font=self.ui_font, font_size=12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            content = VGroup(title, subtitle, rows).arrange(
                DOWN, aligned_edge=LEFT, buff=0.09
            )
        elif algorithm == "dijkstra":
            box = RoundedRectangle(
                width=3.55, height=2.15, corner_radius=0.12, color=GREY_B
            )
            title = Text(
                "Dijkstra Snapshot", font=self.ui_font, font_size=16, weight=BOLD
            )
            subtitle = Text(
                "Choose the node with smallest known g", font=self.ui_font, font_size=12
            )
            rows = VGroup(
                Text("Top: N/A", font=self.ui_font, font_size=12),
                Text("Current g: 0", font=self.ui_font, font_size=12),
                Text("Relax: waiting...", font=self.ui_font, font_size=12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            content = VGroup(title, subtitle, rows).arrange(
                DOWN, aligned_edge=LEFT, buff=0.08
            )
        else:
            box = RoundedRectangle(
                width=3.55, height=2.25, corner_radius=0.12, color=GREY_B
            )
            title = Text("A* Snapshot", font=self.ui_font, font_size=16, weight=BOLD)
            subtitle = Text(
                "Choose the node with smallest f = g + h",
                font=self.ui_font,
                font_size=11,
            )
            rows = VGroup(
                Text("Top: N/A", font=self.ui_font, font_size=12),
                Text("g / h / f: 0 / 0 / 0", font=self.ui_font, font_size=12),
                Text("Update: waiting...", font=self.ui_font, font_size=12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            content = VGroup(title, subtitle, rows).arrange(
                DOWN, aligned_edge=LEFT, buff=0.08
            )

        box.set_fill("#14181F", opacity=0.95)
        self.place_panel_content(box, content, left_pad=0.16, top_pad=0.18)
        box.add(content)
        return box, rows

    def format_frontier(self, algorithm, frontier):
        if algorithm in {"bfs", "dfs"}:
            if not frontier:
                return "[]"
            return "[" + ", ".join(map(str, frontier)) + "]"
        if not frontier:
            return "[]"
        return (
            "["
            + ", ".join(map(str, frontier[:4]))
            + (" ..." if len(frontier) > 4 else "")
            + "]"
        )

    def fit_line(self, content, font_size, max_width, color=WHITE):
        text = Text(content, font=self.ui_font, font_size=font_size, color=color)
        if text.width > max_width:
            text.scale_to_fit_width(max_width)
        return text

    def target_line(self, target, content, font_size, max_width, color=WHITE):
        text = self.fit_line(content, font_size, max_width, color=color)
        text.match_y(target)
        text.align_to(target, LEFT)
        return text

    def summarize_neighbors(self, neighbors, max_items=3):
        if not neighbors:
            return "Checks: none"
        mapped = {"out": "out", "wall": "wall", "seen": "seen", "push": "push"}
        chunks = [
            f"{pos}:{mapped.get(status, status)}"
            for pos, status, _ in neighbors[:max_items]
        ]
        suffix = " ..." if len(neighbors) > max_items else ""
        return "Checks: " + "; ".join(chunks) + suffix

    def build_runtime_panel(self):
        box = RoundedRectangle(
            width=3.55, height=2.05, corner_radius=0.12, color=GREY_B
        )
        box.set_fill("#14181F", opacity=0.95)
        title = Text("Runtime", font=self.ui_font, font_size=16, weight=BOLD)
        current = Text("Current: -", font=self.ui_font, font_size=12)
        frontier = Text("Frontier: []", font=self.ui_font, font_size=12)
        visited = Text("Visited: 0", font=self.ui_font, font_size=12)
        decision = Text("Decision: waiting...", font=self.ui_font, font_size=12)
        checks = Text("Checks: none", font=self.ui_font, font_size=12)
        content = VGroup(title, current, frontier, visited, decision, checks).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08
        )
        self.place_panel_content(box, content, left_pad=0.16, top_pad=0.18)
        box.add(content)
        return box, (current, frontier, visited, decision, checks)

    def construct(self):
        trace, path = search_with_trace(MAZE, self.algorithm)
        rows, cols = len(MAZE), len(MAZE[0])
        cell_size = 0.71
        origin = LEFT * 5.75 + UP * 0.9

        titles = {
            "bfs": ("BFS Maze Search", "FIFO expansion by level"),
            "dfs": ("DFS Maze Search", "LIFO deep-first traversal"),
            "dijkstra": ("Dijkstra Maze Search", "Expand minimum known path cost"),
            "astar": ("A* Maze Search", "Expand minimum f = g + h"),
        }
        title_text, subtitle_text = titles[self.algorithm]
        title = Text(title_text, font=self.ui_font, font_size=31, weight=BOLD)
        subtitle = Text(subtitle_text, font=self.ui_font, font_size=17)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2)
        header.to_edge(UP).shift(RIGHT * 0.2)

        grid_group = VGroup()
        cell_map = {}
        coord_labels = VGroup()

        for r in range(rows):
            for c in range(cols):
                x = origin[0] + c * cell_size
                y = origin[1] - r * cell_size
                center = np.array([x, y, 0])
                base_color = "#22252B" if MAZE[r][c] == 0 else "#111111"
                cell = self.make_cell(center, cell_size, base_color)
                cell_map[(r, c)] = cell
                grid_group.add(cell)

                label = Text(f"{r},{c}", font=self.ui_font, font_size=12, color=GREY_B)
                label.move_to(center)
                coord_labels.add(label)

        start = (0, 0)
        goal = (rows - 1, cols - 1)
        start_marker = Circle(radius=0.18, color=GREEN, fill_opacity=1)
        start_marker.move_to(cell_map[start].get_center())
        goal_marker = Circle(radius=0.18, color=RED, fill_opacity=1)
        goal_marker.move_to(cell_map[goal].get_center())

        grid_title = (
            Text("Maze", font=self.ui_font, font_size=19)
            .next_to(grid_group, UP, buff=0.16)
            .align_to(grid_group, LEFT)
        )
        self.play(
            Write(header), FadeIn(grid_title), Create(grid_group), FadeIn(coord_labels)
        )
        self.play(FadeIn(start_marker), FadeIn(goal_marker))

        legend = VGroup(
            self.make_cell(
                ORIGIN, 0.18, "#2E8B57", stroke_color="#2E8B57", stroke_width=1
            ),
            Text("Visited", font=self.ui_font, font_size=12),
            self.make_cell(
                ORIGIN, 0.18, "#3FA7D6", stroke_color="#3FA7D6", stroke_width=1
            ),
            Text("Frontier", font=self.ui_font, font_size=12),
            self.make_cell(
                ORIGIN, 0.18, "#FFD166", stroke_color="#FFD166", stroke_width=1
            ),
            Text("Current", font=self.ui_font, font_size=12),
            self.make_cell(
                ORIGIN, 0.18, "#F94144", stroke_color="#F94144", stroke_width=1
            ),
            Text("Path", font=self.ui_font, font_size=12),
        ).arrange(RIGHT, buff=0.1)
        legend.to_edge(DOWN).shift(UP * 0.02)
        self.play(FadeIn(legend, shift=UP * 0.2))

        algo_panel, algo_panel_items = self.build_algorithm_panel(self.algorithm)
        metric_box, metric_items = self.build_snapshot_panel(self.algorithm)
        algo_panel.to_edge(RIGHT).shift(UP * 1.25)
        metric_box.next_to(algo_panel, DOWN, buff=0.14).align_to(algo_panel, LEFT)
        self.play(FadeIn(algo_panel), FadeIn(metric_box))

        runtime_box, runtime_items = self.build_runtime_panel()
        runtime_box.next_to(metric_box, DOWN, buff=0.12).align_to(metric_box, LEFT)
        self.play(FadeIn(runtime_box))

        visited_fill = "#2E8B57"
        queued_fill = "#3FA7D6"
        current_fill = "#FFD166"
        path_fill = "#F94144"

        # 初始状态
        self.play(cell_map[start].animate.set_fill(current_fill, opacity=1))

        for idx, step in enumerate(trace):
            current = step["current"]
            neighbors = step["neighbors"]
            frontier = step["frontier"]
            visited = step["visited"]
            extra = step.get("extra", {})

            max_line_w = runtime_box.width - 0.28
            current_desc = f"Current: {current}"
            if self.algorithm in {"dijkstra", "astar"}:
                current_desc += f"  g={extra.get('g', 0):g}"
            if self.algorithm == "astar":
                current_desc += f"  h={extra.get('h', 0):g}  f={extra.get('f', 0):g}"
            current_label = self.target_line(
                runtime_items[0],
                current_desc,
                13,
                max_line_w,
            )
            frontier_label = self.target_line(
                runtime_items[1],
                f"Frontier: {self.format_frontier(self.algorithm, frontier)}",
                13,
                max_line_w,
            )
            visited_label = self.target_line(
                runtime_items[2],
                f"Visited: {len(visited)}",
                13,
                max_line_w,
            )
            decision_label = self.target_line(
                runtime_items[3],
                f"Decision: {step['decision']}",
                13,
                max_line_w,
            )
            checks_label = self.target_line(
                runtime_items[4],
                self.summarize_neighbors(neighbors),
                13,
                max_line_w,
                color=BLUE_C,
            )

            self.play(
                Transform(runtime_items[0], current_label),
                Transform(runtime_items[1], frontier_label),
                Transform(runtime_items[2], visited_label),
                Transform(runtime_items[3], decision_label),
                Transform(runtime_items[4], checks_label),
                cell_map[current].animate.set_fill(current_fill, opacity=1),
                run_time=0.45,
            )

            self.play(Indicate(cell_map[current], color=YELLOW), run_time=0.35)

            if self.algorithm in {"bfs", "dfs"}:
                frontier_item = self.target_line(
                    metric_items[0],
                    f"Frontier: {self.format_frontier(self.algorithm, frontier)}",
                    13,
                    metric_box.width - 0.28,
                )
                action_item = self.target_line(
                    metric_items[1],
                    (
                        "Action: pop from queue"
                        if self.algorithm == "bfs"
                        else "Action: pop from stack"
                    ),
                    13,
                    metric_box.width - 0.28,
                )
                next_node = frontier[0] if frontier else None
                next_item = self.target_line(
                    metric_items[2],
                    f"Next: {next_node}" if next_node else "Next: N/A",
                    13,
                    metric_box.width - 0.28,
                )
                self.play(
                    Transform(metric_items[0], frontier_item),
                    Transform(metric_items[1], action_item),
                    Transform(metric_items[2], next_item),
                    run_time=0.25,
                )
            elif self.algorithm == "dijkstra":
                top = frontier[0] if frontier else None
                top_item = self.target_line(
                    metric_items[0],
                    f"Top: {top}" if top else "Top: N/A",
                    13,
                    metric_box.width - 0.28,
                )
                g_item = self.target_line(
                    metric_items[1],
                    f"Current g: {extra.get('g', 0):g}",
                    13,
                    metric_box.width - 0.28,
                )
                relax_text = "Relaxation: compare nd with dist[n]"
                relax_item = self.target_line(
                    metric_items[2], relax_text, 13, metric_box.width - 0.28
                )
                self.play(
                    Transform(metric_items[0], top_item),
                    Transform(metric_items[1], g_item),
                    Transform(metric_items[2], relax_item),
                    run_time=0.25,
                )
            else:
                top = frontier[0] if frontier else None
                top_item = self.target_line(
                    metric_items[0],
                    f"Top: {top}" if top else "Top: N/A",
                    13,
                    metric_box.width - 0.28,
                )
                ghf_item = self.target_line(
                    metric_items[1],
                    f"{extra.get('g', 0):g} / {extra.get('h', 0):g} / {extra.get('f', 0):g}",
                    13,
                    metric_box.width - 0.28,
                )
                update_item = self.target_line(
                    metric_items[2],
                    "Update: tentative_g + heuristic",
                    13,
                    metric_box.width - 0.28,
                )
                self.play(
                    Transform(metric_items[0], top_item),
                    Transform(metric_items[1], ghf_item),
                    Transform(metric_items[2], update_item),
                    run_time=0.25,
                )

            for neighbor, status, reason in neighbors:
                if status == "push":
                    self.play(
                        cell_map[neighbor].animate.set_fill(queued_fill, opacity=1),
                        run_time=0.18,
                    )

            # 标记当前格子为已访问，避免与后续当前格混淆
            if current != goal:
                self.play(
                    cell_map[current].animate.set_fill(visited_fill, opacity=1),
                    run_time=0.18,
                )

        if path:
            result_text = self.target_line(
                runtime_items[4],
                "Checks: path found",
                13,
                runtime_box.width - 0.28,
                color=GREEN,
            )
            self.play(Transform(runtime_items[4], result_text))

            path_lines = VGroup()
            for a, b in zip(path[:-1], path[1:]):
                line = Line(
                    cell_map[a].get_center(),
                    cell_map[b].get_center(),
                    color=path_fill,
                    stroke_width=8,
                )
                path_lines.add(line)
            self.play(Create(path_lines), run_time=1.2)
            for node in path:
                self.play(
                    cell_map[node].animate.set_fill(path_fill, opacity=1), run_time=0.12
                )

        else:
            result_text = self.target_line(
                runtime_items[4],
                "Checks: no valid path",
                13,
                runtime_box.width - 0.28,
                color=RED,
            )
            self.play(Transform(runtime_items[4], result_text))

        self.wait(2)


class BFSMazeSearchScene(MazeSearchScene):
    algorithm = "bfs"


class DFSMazeSearchScene(MazeSearchScene):
    algorithm = "dfs"


class DijkstraMazeSearchScene(MazeSearchScene):
    algorithm = "dijkstra"


class AStarMazeSearchScene(MazeSearchScene):
    algorithm = "astar"
