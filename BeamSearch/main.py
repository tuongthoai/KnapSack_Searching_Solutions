# N items each has v, w, c by order is value, weight, and type
# W is maximum weight allow
# Find the largest value with each class at least 1
import heapq
import random
import time
import numpy as np


class BeamSearch:
    def __init__(self, file_name: str):
        self.input_file_name = file_name
        self.N = 0
        self.W = 0

        # Read Input
        f = open(file_name, "r")
        self.W = float(f.readline())
        self.N_type = int(f.readline())
        self.weights = tuple(list(map(float, f.readline().strip().split(','))))
        self.values = tuple(list(map(float, f.readline().strip().split(','))))
        self.types = tuple(list(map(float, f.readline().strip().split(','))))
        f.close()

        self.N = len(self.values)
        self.MAX_BEAM_SIZE = 100
        self.ALPHA = min(self.values)
        self.MINIMUM_VALUE = min(self.values)
        self.MAX_RESTART_CNT = 10

        self.value_by_weights = [tuple([val / self.weights[i], i]) for i, val in enumerate(self.values)]
        self.value_by_weights.sort(key=lambda k: k[0], reverse=True)

        self.vbw_dict = dict()

        for i in range(1, self.N_type + 1):
            self.vbw_dict[i] = []

        for i, val in enumerate(self.value_by_weights):
            var = self.vbw_dict[self.types[val[1]]]
            var.append(val[1])

    def evaluate_profit(self, x: np.array) -> float:
        total_value = np.dot(x, self.values)
        total_weight = np.dot(x, self.weights)
        number_of_type = set([self.types[i] * val for i, val in enumerate(x)])
        number_of_type.remove(0)
        if total_weight > self.W:
            return -1

        if total_weight + self.MINIMUM_VALUE > self.W and len(number_of_type) < self.N_type:
            return 0
        else:
            return total_value - self.ALPHA * (self.N_type - len(number_of_type))

    # INPUT: a np.array OUTPUT list[np.array]
    def gen_neighborhood(self, cur_x: np.array) -> list:
        list_x = list(cur_x)
        neighbors = []
        for i in range(self.N):
            neighbors.append(list_x.copy())
            neighbors[i][i] = 1 - neighbors[i][i]
        return neighbors

    # create the initial solution
    def initial_solution(self):
        x = [0] * self.N
        x_weight = 0
        used = set()
        ids = np.random.randint(0, self.N - 1, size=self.N)
        for i in range(self.N):
            if ids[i] in used:
                continue

            used.add(ids[i])
            if x_weight + self.weights[ids[i]] < self.W:
                x[ids[i]] = 1
                x_weight += self.weights[ids[i]]
            else:
                break

        return x

    def initBeam(self):
        current_solution = self.initial_solution()  # x_curr will hold the current solution
        current_profit = self.evaluate_profit(
            current_solution)  # f_curr will hold the evaluation of the current solution
        return current_solution, current_profit

    def local_search_best_improvement(self, cur_x: np.array, cur_profit: float) -> list[np.array]:
        neighborhood = self.gen_neighborhood(cur_x)  # create a list of all neighbors in the neighborhood of x_curr
        new_improvements = []
        for neighbor in neighborhood:  # evaluate every member in the neighborhood of x_curr
            value = self.evaluate_profit(neighbor)
            if value >= cur_profit:
                new_improvements.append(tuple([value, neighbor]))
        return new_improvements

    def initBeam2(self, x):
        new_x = x.copy()
        random.shuffle(new_x)
        MAX_RETRIES = 100
        tries = 0
        while self.evaluate_profit(new_x) < 0 and tries < MAX_RETRIES:
            random.shuffle(new_x)
            tries += 1

        if tries == MAX_RETRIES:
            return self.initBeam()
        else:
            return new_x, self.evaluate_profit(new_x)

    def solve(self):
        # prepare process
        x_curr = []
        f_curr = []
        x_best = None  # best configuration so far
        f_best = -1  # best value so far

        x = [0] * self.N
        w = 0
        v = 0
        # this selection the top highest ratio of value to weight
        # this gready initial chosen state is a very good initial state following
        # Discrete Variable Extremum Problems. Dantzig, G.B. 1975

        # this part ensure the inital x has enough class
        for i in range(1, self.N_type + 1):
            var = self.vbw_dict[i]
            for id in var:
                if w + self.weights[id] < self.W:
                    x[id] = 1
                    w += self.weights[id]
                    v += self.values[id]
                    break

        # greedily add items into x
        for _, val in enumerate(self.value_by_weights):
            if w + self.weights[val[1]] < self.W and x[val[1]] == 0:
                x[val[1]] = 1
                w += self.weights[val[1]]
                v += self.values[val[1]]
            else:
                break

        x_curr.append(tuple(x))
        f_curr.append(v)
        if f_best < v:
            f_best = v
            x_best = x

        # this part generate beam follow by the initial configuration
        for _ in range(self.MAX_BEAM_SIZE - 1):
            new_x, val = self.initBeam2(x)
            x_curr.append(new_x)
            f_curr.append(self.evaluate_profit(new_x))
            if f_curr[len(f_curr) - 1] > f_best:
                f_best = f_curr[len(f_curr) - 1]
                x_best = new_x

        # starting to solve
        while True:
            new_improvements = []
            # Collect Improvements
            for i, x in enumerate(x_curr):
                improvements = self.local_search_best_improvement(x, f_curr[i])
                new_improvements.extend(improvements)

            # Get the top improvements
            top_improvements = heapq.nlargest(self.MAX_BEAM_SIZE, new_improvements, key=lambda k: k[0])

            # Case when no new improvements found
            if not top_improvements:
                break

            # Handles the common case when there is an improved solution for each neighbor
            elif len(top_improvements) == self.MAX_BEAM_SIZE:
                for i, solution in enumerate(top_improvements):
                    x_curr[i] = solution[1]
                    f_curr[i] = solution[0]

            # This is for the end when the number of solutions < number of beams.
            else:
                length = len(top_improvements)
                for beam in range(length):
                    x_curr[beam] = top_improvements[beam][1]  # access to nparray
                    f_curr[beam] = top_improvements[beam][0]  # access to profit
                # For any excess beams, assign the first improvement.
                for beam in range(length, self.MAX_BEAM_SIZE):
                    x_curr[beam] = top_improvements[0][1]
                    f_curr[beam] = top_improvements[0][0]

            if f_best < top_improvements[0][0]:
                f_best = top_improvements[0][0]
                x_best = top_improvements[0][1]

        return f_best, x_best


# Utility function
def getInputName(cnt: int) -> str:
    return f"INPUT_{cnt}.txt"


def getOutputName(cnt: int) -> str:
    return f"OUTPUT_{cnt}.txt"


def write_array(f, data):
    n = len(data)
    f.write(str(data[0]))
    for i in range(1, n):
        f.write(f", {data[i]}")
    f.write('\n')


def main():
    for test_case in range(1, 11):
        file_path = "./TEST_CASE/" + getInputName(test_case)
        fout = open("./RESULT/" + getOutputName(test_case), 'w')

        t1 = time.time()
        solver = BeamSearch(file_path)
        answer_profit, answer_cfg = solver.solve()
        t2 = time.time() - t1
        fout.write(f"{answer_profit}\n")
        write_array(fout, answer_cfg)
        print("RUNTIME: {}".format(t2))
        fout.close()


if __name__ == '__main__':
    main()