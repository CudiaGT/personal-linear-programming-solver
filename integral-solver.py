import sympy, math
import matplotlib.pyplot as plt

def solve(lp):
    lp['outcome'] = 'unsolved'
    find_feasible_basis(lp)
    if lp['outcome'] == 'infeasible':
        return
    while lp['outcome'] == 'unsolved':
        improve_current_basis(lp)
    if lp['outcome'] == 'minimized':
        compute_shadow_prices(lp)

def compute_dictionary(lp):
    A = lp['A']
    m, n = A.shape

    # keep B sorted
    lp['B'] = list(sorted(lp['B']))

    # get N
    lp['N'] = list(set(range(A.cols)) - set(lp['B']))

    # compute necessary ingredients for dictionary
    A_B = A.extract(range(m), lp['B'])
    A_N = A.extract(range(m), lp['N'])
    c_B = lp['c'].extract(lp['B'], range(1))
    c_N = lp['c'].extract(lp['N'], range(1))

    # set current basic solution
    x_B = A_B.inv() * lp['b']
    solution = sympy.zeros(n,1)
    for i in range(len(lp['B'])):
        solution[lp['B'][i]] = x_B[i]
    lp['solution'] = solution
    lp['cost'] = solution.dot(lp['c'])

    # save dictionary and rcv
    lp['dictionary'] = -A_B.inv() * A_N
    lp['rcv'] = c_N - (A_B.inv() * A_N).T * c_B

# helper function to find entering and exiting variables
def enter_exit(lp):
    lp['enter'] = None
    lp['exit'] = None

    # entering variable
    for i in range(lp['rcv'].shape[0]):
        if lp['rcv'][i] < 0:
            lp['enter'] = i
            break
    if lp['enter'] is None:
        return
    else:
        # exiting variable
        allowable_increase = float('inf')
        for j in range(len(lp['B'])):
            coeff = lp['dictionary'][j, lp['enter']]
            if coeff < 0:
                ratio = -lp['solution'][lp['B'][j]] / coeff
                if ratio < allowable_increase:
                    allowable_increase = ratio
                    lp['exit'] = j

def improve_current_basis(lp):
    # get enter, exit
    enter_exit(lp)

    # if no entering var, optimal (minimized)
    if lp['enter'] is None:
        lp['outcome'] = 'minimized'

    # if entering var and no exiting var, unbounded
    elif lp['exit'] is None:
            lp['outcome'] = 'unbounded'

    # otherwise, pivot and compute new dictionary
    else:
        lp['B'].remove(lp['B'][lp['exit']])
        lp['B'].append(lp['N'][lp['enter']])
        compute_dictionary(lp)

def compute_shadow_prices(lp):
    A = lp['A']
    m, n = A.shape
    A_B = A.extract(range(m), lp['B'])
    c_B = lp['c'].extract(lp['B'], range(1))
    lp['marginals'] = A_B.T.inv() * c_B

# helper function to perform move 1 of FPLP
def move1(FPLP):
    m, n = FPLP['A'].shape
    FPLP['enter'] = None

    # find exit var (i.e. artificial var)
    for i in range(len(FPLP['B'])):
        if FPLP['B'][i] >= n - FPLP['num_artificial']:
            FPLP['exit'] = i

    # find enter var
    for j in range(FPLP['rcv'].rows):
        if FPLP['N'][j] < n - FPLP['num_artificial'] and FPLP['dictionary'][i, j] != 0:
            FPLP['enter'] = j
    if FPLP['enter'] == None:
        return

    # pivot
    FPLP['B'].remove(FPLP['B'][FPLP['exit']])
    FPLP['B'].append(FPLP['N'][FPLP['enter']])

    # each time move 1 is performed, compute new dictionary
    compute_dictionary(FPLP)

# helper function to perform move 2 of FPLP
def move2(FPLP):
    m, n = FPLP['A'].shape
    FPLP['redundant'] = []
    # locate first/only artificial var
    for i in range(len(FPLP['B'])):
        if FPLP['B'][i] >= n - FPLP['num_artificial']:
            FPLP['redundant'].append(i)

    # remove appropriate entries/rows from A, b, c
    FPLP['A'].row_del(i)
    FPLP['b'].row_del(i)
    FPLP['c'].row_del(i)

    # remove appropriate column from A and current basis (i.e. remove artificial var)
    FPLP['A'].col_del(n - FPLP['num_artificial'] + i)
    FPLP['B'].remove(FPLP['B'][i])

# helper function to remove artificial variables from B
def remove_artificial(FPLP):
    m, n = FPLP['A'].shape

    # while there are artificial variables in optimal basis, perform as many move 1 as possible
    while any(var > n - FPLP['num_artificial'] for var in FPLP['B']):
        move1(FPLP)
        # if move 1 cannot be performed, perform move 2
        if FPLP['enter'] is None:
            move2(FPLP)

def find_feasible_basis(lp):

    # initialize a separate dictionary for FPLP to prevent mutating the original LP
    FPLP = {}
    FPLP['A'] = lp['A']
    FPLP['b'] = lp['b']
    m, n = lp['A'].shape
    FPLP['num_artificial'] = m

    # check if b has any negative values. if so, negate the appropriate line(s) of A and b
    for i in range(len(FPLP['b'])):
        if FPLP['b'][i] < 0:
            FPLP['b'][i] = -FPLP['b'][i]
            FPLP['A'][i, :] = -FPLP['A'].row(i)

    # add artificial variables to 'A' using identity matrix
    I = sympy.eye(m)
    FPLP['A'] = sympy.Matrix.hstack(FPLP['A'], I)

    # use sum of aritifical variables as objective function
    FPLP['c'] = sympy.Matrix([0] * n + [1] * m)

    # use artificial variables as starting feasible basis for FPLP
    FPLP['B'] = list(range(n, n + m))

    # first, solve FPLP once to check for feasibility of original LP
    FPLP['outcome'] = 'unsolved'
    compute_dictionary(FPLP)
    while FPLP['outcome'] == 'unsolved':
        improve_current_basis(FPLP)

    # check for feasibility of original lp
    if FPLP['cost'] != 0:
        lp['outcome'] = 'infeasible'
        return

    # if the original LP was identified to be feasible, continue Phase 1 to find starting feasible basis for original lp
    remove_artificial(FPLP)

    # if and only if move 2 was performed, modify A and b of original lp
    if 'redundant' in FPLP:
        for i in range(len(FPLP['redundant'])):
            lp['A'].row_del(FPLP['redundant'][i])
        lp['b'] = FPLP['b']

    # when Phase 1 is complete, pass the optimal basis of FPLP as starting feasible basis for the original lp
    lp['B'] = FPLP['B']
    compute_dictionary(lp)

########## Additional Features for Solving ILP ##########

# helper function to check integrality of a given number
def is_integer(num):
    return num % 1 == 0

# helper function to check integrality of a solution
def is_integral(solution):
    for entry in solution:
        if not is_integer(entry):
            return False
    return True

def solve_ilp(ilp):
    m, n = ilp['A'].shape
    final_list = []
    lowest_cost = float('inf') # lowest cost (with integral solutions)
    solve(ilp) # solve relaxation
    # if relaxation does not meet the integral constraint, perform branch and bound
    if not is_integral(ilp['solution']):
        ilp['ilp_cost'], ilp['ilp_solution'], final_list = branch_and_bound(ilp, lowest_cost)
        ilp['ilp_solution'] = ilp['ilp_solution'][:n, :]
        ilp['ilp_subproblems'] = len(final_list)
    else:
        ilp['ilp_cost'], ilp['ilp_solution'], ilp['subproblems'], final_list = ilp['cost'], ilp['solution'], 0, final_list
    return final_list

def branch_and_bound(ilp, lowest_cost):
    bfs_queue = []
    current_level = []
    ilp['y'] = 0
    ilp['level'] = 0
    queue_branches(ilp, bfs_queue)
    i = 0
    nodes_in_current_level = 2

    # while there are uncomputed branches in the queue
    while i < len(bfs_queue):
        # choose the next branch
        current_node = bfs_queue[i]
        current_level.append(current_node)
        # obtain solution of the branch via dual simplex method
        compute_dictionary(current_node)
        solve_dual_simplex(current_node)
        # if the solution is integral, update the lowest_cost
        if is_integral(current_node['solution']) and current_node['cost'] <= lowest_cost:
            lowest_cost = current_node['cost']
            solution_with_lowest_cost = current_node['solution']
            current_node['integral'] = True
        # if all the nodes in current level has been computed, append appropriate nodes to the bfs_queue
        if i == nodes_in_current_level - 1:
            nodes_appended = 0
            for node in current_level:
                if node['outcome'] != 'infeasible' and 'integral' not in node and node['cost'] <= lowest_cost:
                    queue_branches(node, bfs_queue)
                    nodes_appended += 2
            # update how many nodes will be in the next level
            nodes_in_current_level += nodes_appended
            # reset all the variables and objects relative to current level
            current_level = []
        i += 1
    bfs_queue.insert(0, ilp)
    return lowest_cost, solution_with_lowest_cost, bfs_queue

# helper function to create two branches (north LP and south LP) and queue them to be solved in BFS order
def queue_branches(ilp, bfs_queue):
    m, n = ilp['A'].shape
    b_var = 0 # index of the branching variable
    while b_var < len(ilp['solution']):
        if not is_integer(ilp['solution'][b_var]):
            break
        b_var += 1
    if b_var == len(ilp['solution']):
        return

    # define north (greater than) LP
    gt_lp = {}
    slack_cost = sympy.Matrix([0])
    branch_row = sympy.zeros(1, n)
    branch_row[b_var] = 1
    slack_col = sympy.zeros(m + 1, 1)
    slack_col[-1] = -1

    # conversion to sympy object type and then ceiling the value
    new_b_val = sympy.S(ilp['solution'][b_var])
    ceiling = sympy.ceiling(new_b_val)
    new_b = sympy.Matrix([ceiling])

    gt_lp['A'] = sympy.Matrix.vstack(ilp['A'], branch_row) # append new row to A (branch constraint)
    gt_lp['A'] = gt_lp['A'].row_join(slack_col) # append new column to A (slack variable to handle branch constraint)
    gt_lp['c'] = ilp['c'].T.row_join(slack_cost).T # new cost
    gt_lp['b'] = ilp['b'].T.row_join(new_b).T # new b
    gt_lp['B'] = ilp['B']
    gt_lp['B'].append(n)
    gt_lp['parent_y'] = ilp['y']
    gt_lp['parent_x'] = ilp['cost']
    gt_lp['level'] = ilp['level'] + 1
    gt_lp['y'] = gt_lp['parent_y'] + (1/2) ** ilp['level']

    # define south (less than) LP
    lt_lp = {}
    slack_col[-1] = 1

    # conversion to sympy object type, and then ceiling the value
    new_b_val = sympy.S(ilp['solution'][b_var])
    floored = sympy.floor(new_b_val)
    new_b = sympy.Matrix([floored])

    lt_lp['A'] = sympy.Matrix.vstack(ilp['A'], branch_row)
    lt_lp['A'] = lt_lp['A'].row_join(slack_col)
    lt_lp['c'] = ilp['c'].T.row_join(slack_cost).T
    lt_lp['b'] = ilp['b'].T.row_join(new_b).T
    lt_lp['B'] = gt_lp['B']
    lt_lp['parent_y'] = ilp['y']
    lt_lp['parent_x'] = ilp['cost']
    lt_lp['level'] = ilp['level'] + 1
    lt_lp['y'] = gt_lp['parent_y'] - (1/2) ** ilp['level']

    bfs_queue.append(gt_lp)
    bfs_queue.append(lt_lp)

def enter_exit_dual(lp):
    m, n = lp['A'].shape
    lp['exit'], lp['enter'] = None, None
    for i in range(len(lp['solution'])):
        if lp['solution'][i] < 0:
            lp['exit'] = lp['B'].index(i)
            break
    if lp['exit'] is None:
        return
    min_ratio = float('inf')
    for j in range(len(lp['N'])):
        # ratio = abs(rcv(i) / dictionary_coeff(i))
        if lp['dictionary'][lp['exit'], j] > 0:
            ratio = abs(lp['rcv'][j] / lp['dictionary'][lp['exit'], j])
            if ratio < min_ratio:
                min_ratio = ratio
                lp['enter'] = j

def solve_dual_simplex(lp):
    lp['outcome'] = 'unsolved'
    while lp['outcome'] == 'unsolved':
        compute_dictionary(lp)
        enter_exit_dual(lp)
        if lp['exit'] is None:
            lp['outcome'] = 'minimized'
            return
        elif lp['enter'] is None:
            lp['outcome'] = 'infeasible'
            return
        else:
            lp['B'].remove(lp['B'][lp['exit']])
            lp['B'].append(lp['N'][lp['enter']])

########## Additional Features for Plotting ##########

def draw_bb_tree(ilp):
    lp_list = []
    lp_list = solve_ilp(ilp)
    for i in range(len(lp_list)):
        lp = lp_list[i]
        if lp['outcome'] == 'infeasible':
            lp['x'] = lp['parent_x'] + 1
            plt.scatter(lp['x'], lp['y'], color='black', s=150, zorder=2)
        if 'integral' in lp:
            lp['x'] = lp['cost']
            plt.scatter(lp['x'], lp['y'], color='green', s=150, zorder=2)
        elif lp['outcome'] == 'minimized':
            lp['x'] = lp['cost']
            plt.scatter(lp['x'], lp['y'], color='red', s=150, zorder=2)
        if lp['level'] != 0:
            plt.plot([lp['x'], lp['parent_x']], [lp['y'], lp['parent_y']], color='gray', zorder=1)
    plt.plot()