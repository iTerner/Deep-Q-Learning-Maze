# Random Maze Generator using Depth-first Search
import random
import matplotlib.pyplot as plt
import numpy as np

# the name for the maze
SAVE_NAME = "maze_generator/test_maze"

# width and height of the Maze
MX = 20
MY = 20

# initialize the maze
maze = [[0 for _ in range(MX)] for _ in range(MY)]

# 4 directions to move in the
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

# RGB colors for the maze
colors = [(0, 0, 0), (255, 255, 255)]

# start the maze from a random cell
cx, cy = random.randint(0, MX - 1), random.randint(0, MY - 1)
maze[cx][cy] = 1
stack = [(cx, cy, 0)]  # stack elements: (pos_x, pos_y, direction)

while len(stack) > 0:
    (cx, cy, cd) = stack[-1]
    # to prevent zigzags:
    # if changed direction in the last move, then he cannot change again
    if (len(stack)) > 2:
        if cd != stack[-2][2]:
            dir_range = [cd]
        else:
            dir_range = range(4)
    else:
        dir_range = range(4)

    # find a new cell to add
    nlst = []
    for i in dir_range:
        nx = cx + dx[i]
        ny = cy + dy[i]
        if nx >= 0 and nx < MX and ny >= 0 and ny < MY:
            if maze[nx][ny] == 0:
                ctr = 0  # of occupied neighbors must be 1
                for j in range(4):
                    ex = nx + dx[j]
                    ey = ny + dy[j]
                    if ex >= 0 and ex < MX and ey >= 0 and ey < MY:
                        if maze[ex][ey] == 0:
                            ctr += 1
                if ctr == 1:
                    nlst.append(i)

    # if 1 or more neighbors available then randomly select one and move
    if len(nlst) > 0:
        ir = nlst[random.randint(0, len(nlst) - 1)]
        cx += dx[ir]
        cy += dy[ir]
        maze[cx][cy] = 1
        stack.append((cx, cy, ir))
    else:
        stack.pop()

maze = np.array(maze)
maze -= 1
maze = abs(maze)

maze[0][0] = 0
maze[MX - 1][MY - 1] = 0

np.save(SAVE_NAME, np.array(maze))
