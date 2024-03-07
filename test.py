from collections import defaultdict
import numpy as np

moves = [(1, 2), (2, 1), (3, 3)]

returns = {move: [] for move in moves}
#print(returns)

print(returns[(1,2)])

returns[(1,2)].append(25)
returns[(1,2)].append(15)

print(returns[(1,2)])
