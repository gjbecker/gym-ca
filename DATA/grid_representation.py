import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCALE = {
        'empty': 0,
        'ego agent': 1,
        'other agents': 2,
        'ego goal': 3
    }

def draw(grid, RES, side, x_coord, y_coord, rad, value, print_grid=False):
    # Coordinate transformation
    cx = (side+x_coord) * ((RES-1))/(side*2)
    cy = (side-y_coord) * ((RES-1))/(side*2)
    r = rad*((RES-1))/(side*2)

    if cx > RES-1 or cy > RES-1:
        # print(f'{x_coord,y_coord} outside coordinate range!')
        return grid

    for angle in np.arange(0, 2*np.pi, np.pi/60):
        x = int(round(cx + r*np.cos(angle)))
        y = int(round(cy + r*np.sin(angle)))
        # print(x,y, angle)
        x = max(min(RES-1, x), 0)
        y = max(min(RES-1, y), 0)

        # Fill circle
        if np.pi/2 >= angle >= 0:
            for i in range(int(cx), x+1):
                for j in range(int(cy), y+1):
                    grid[j,i] = value

        elif np.pi >= angle >= np.pi/2:
            for i in range(x, int(cx)+1):
                for j in range(int(cy), y+1):
                    grid[j,i] = value

        elif 3*np.pi/2 >= angle >= np.pi:
            for i in range(x, int(cx)+1):
                for j in range(y, int(cy)+1):
                    grid[j,i] = value

        elif 2*np.pi >= angle >= 3*np.pi/2:
            for i in range(int(cx), x+1):
                for j in range(y, int(cy)+1):
                    grid[j,i] = value        

    if print_grid:
        print(grid)
        print('='*50)
    return grid


def episode_grid(episode, RES=512, side=7.5, agent_r=2, goal_r=0.2, step_range=None, plot=False):
    agents = np.arange(len(episode['radii']))
    radii = episode['radii']
    goals = episode['goals']
    states = episode['states']
    actions = episode['actions']
    end = len(episode['states'][0])
    ego = 0
    if step_range != None:
        assert(type(step_range) == list)
        assert(len(step_range) == 2)
        steps = range(step_range[0], step_range[1])
    else:
        steps = range(episode['steps'])

    # Draw goal for ego agent
    empty_grid = np.zeros((RES,RES))
    grid_goal = draw(np.copy(empty_grid), RES, side, goals[ego][0], goals[ego][1], goal_r, SCALE['ego goal'])
    obs = []
    for step in steps:
        grid = np.copy(grid_goal)
        head_grid = np.copy(empty_grid)
        speed_grid = np.copy(empty_grid)
        # Draw positions for agents
        for agent in agents:
            if agent == ego:
                grid = draw(grid, RES, side, states[agent][step][0], states[agent][step][1], radii[agent], SCALE['ego agent'])
                heading = draw(head_grid, RES, side, states[agent][step][0], states[agent][step][1], radii[agent], states[agent][step][2])   # heading channel for ego agent only
            else:
                grid = draw(grid, RES, side, states[agent][step][0], states[agent][step][1], radii[agent], SCALE['other agents'])
            speed = draw(speed_grid, RES, side, states[agent][step][0], states[agent][step][1], radii[agent], actions[agent][step][0])    # draw speeds for all agents
        obs.extend([[np.copy(grid), np.copy(speed), np.copy(heading)]])
        step += 1
        if plot and (step % 10 == 0 or step == 1 or step == end):
            plt.imshow(grid, cmap=cm.gray)
            plt.title('Step: ' + str(step))
            plt.show()
    # print(f'STEP: {step}, OBS SHAPE: {np.array(obs).shape}')
    # print(f'OBS RESHAPE: {np.array(obs).reshape((-1,3,RES,RES)).shape}')
    return np.array(obs)

def step_grid(data, RES=512, side=7.5, agent_r=2, goal_r=0.2, plot=False):
    agents = np.arange(len(data['radii']))
    radii = data['radii']
    goals = data['goals']
    states = data['states']
    actions = data['actions']
    ego = 0
    empty_grid = np.zeros((RES,RES))
    grid = draw(np.copy(empty_grid), RES, side, goals[ego][0], goals[ego][1], goal_r, SCALE['ego goal'])
    for agent in agents:
            if agent == ego:
                grid = draw(grid, RES, side, states[agent][0], states[agent][1], radii[agent], SCALE['ego agent'])
                heading = draw(empty_grid, RES, side, states[agent][0], states[agent][1], radii[agent], states[agent][2])   # heading channel for ego agent only
            else:
                grid = draw(grid, RES, side, states[agent][0], states[agent][1], radii[agent], SCALE['other agents'])
            speed = draw(empty_grid, RES, side, states[agent][0], states[agent][1], radii[agent], actions[agent][0])    # draw speeds for all agents

    return np.array([[np.copy(grid), np.copy(speed), np.copy(heading)]])