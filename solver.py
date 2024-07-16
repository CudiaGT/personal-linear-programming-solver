import sympy, math

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