import heapq
import time
import psutil
def Branch_And_Bound(W, m, weights, values, classes):
    n = len(weights)
    root = (0, set(), 0, 0, [])
    pq = [(0, root)]
    best_value = 0
    while pq:
        _, node = heapq.heappop(pq)
        level, selected_classes, value, weight, selected_items = node
    # If we have explored all items, update the best solution found so far
        if (level == n):
            if (value > best_value):
                best_value = value
                best_selected = selected_items
            continue
        item_weight, item_value, item_class = weights[level], values[level], classes[level]
        if item_class not in selected_classes:
            new_covered_classes = selected_classes
            new_covered_classes.add(item_class)
        upper_bound = value + (W - weight) * (item_value / item_weight)
        if (weight + item_weight <= W):
            heapq.heappush(pq, (-upper_bound, (level + 1, new_covered_classes, value + item_value, weight + item_weight, selected_items + [1])))
            heapq.heappush(pq, (-upper_bound, (level + 1, new_covered_classes, value, weight, selected_items + [0])))
        else:
            heapq.heappush(pq, (0, (level + 1, new_covered_classes, value, weight, selected_items + [0])))
    return best_value, best_selected

start_time = time.time()
proccess = psutil.Process()  
# Read input from file
with open('Branch&Bound./TEST_CASE./INPUT_2.txt', 'r') as f:
    W = float(f.readline())   # Knapsack capacity
    m = int(f.readline())     # Number of classes
    weights = list(map(float, f.readline().strip().split(',')))   
    values = list(map(int, f.readline().strip().split(',')))      
    classes = list(map(int, f.readline().strip().split(',')))     
    f.close()

best_value, best_selected = Branch_And_Bound(W, m, weights, values, classes)

# Write output to file
with open('Branch&Bound./RESULT./OUTPUT_2.txt', 'w') as f:
    f.write(str(best_value) + '\n')
    f.write(' '.join(map(str, best_selected)) + '\n')
    end_time = time.time()
    f.close()
print("Time run: "+ str((end_time-start_time)*1000)+ "ms")
print("Memory used: "+ str(proccess.memory_info().rss/1024/1024)+" MB")