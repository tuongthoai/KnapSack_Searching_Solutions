import heapq, itertools, time, psutil
# It's a class that represents an item in the knapsack problem. It has three attributes: weight, value, and typeClass
class Item:
    def __init__(self, _weight: float, _val: int, _typeclass: int) -> None:
        self.value = _val
        self.weight = _weight
        self.typeClass = _typeclass
        
    def __str__(self) -> str:
        return f"[Weight: {self.weight}, Value: {self.value}, Class: {self.typeClass}]"
    
# A bag is a list of items, a weight, a value, and a list of the number of items of each class
class Bag:
    def __init__(self, _items: list[Item], _val: int, _weight: float, _classList: list[int]) -> None:
        self.weight = _weight
        self.value = _val
        self.ItemList = _items
        self.ClassList = _classList
        
    def __hash__(self):
        item_list_hashes = [hash(item) for item in self.ItemList]
        return hash((tuple(item_list_hashes), self.weight, self.value))
    
    def __str__(self) -> str:
        return f"[Weight: {self.weight}, Value: {self.value}, Item list: {', '.join(str(i) for i in self.ItemList)}]"
    
    def valueBag(self):
        for i in self.ClassList:
            if i == 0: return 0
        return self.value
    
    def __lt__(self, other):
        return self.valueBag() > other.valueBag() 
        
    def add_item(self, item: Item):
        if Item in self.ItemList: return
        self.ItemList.append(item)
        self.weight += item.weight
        self.value += item.value
        self.ClassList[item.typeClass - 1] += 1

def readInputFile(indexFile: int):
    """
    It reads the input file and returns the weight of the bag, the number of classes, and a list of
    items
    
    :param indexFile: the index of the input file
    :type indexFile: int
    :return: weightBag, numClass, ItemList
    """
    name = "./TEST_CASE/INPUT_" + str(indexFile) + ".txt"
    ItemList = list()
    f = open(name,'r')
    weightBag = float(f.readline())
    numClass = int(f.readline())
    weightList = list(f.readline().split(','))
    valList = list(f.readline().split(','))
    classList = list(f.readline().split(','))
    for i in range(len(weightList)): ItemList.append(Item(float(weightList[i]), int(valList[i]), int(classList[i])))
    f.close()
    return weightBag, numClass, ItemList

def writeOutputFile(indexFile: int, solution, ItemList):
    """
    It writes the solution to the output file
    
    :param indexFile: the index of the file to be read
    :type indexFile: int
    :param solution: the solution object returned by the solver
    :param ItemList: a list of all the items in the problem
    """
    name = "./RESULT/OUTPUT_" + str(indexFile) + ".txt"
    f = open(name, 'w')
    f.write(str(solution.value))
    f.write('\n')
    for i in ItemList:
        if i in solution.ItemList: f.write('1 ')
        else: f.write('0 ')

# It's a function that takes the weight of the bag, the number of classes, and a list of items as input. It returns the best bag that has all the classes.
def bruteForce(weightBag: float, numClass: int, itemsList: list[Item]):
    """
    For every possible combination of items, create a bag and add items to it until it's full. If the
    bag is full, add it to the list of bags. Then, find the bag with the highest value
    
    :param weightBag: the maximum weight of the bag
    :type weightBag: float
    :param numClass: number of classes
    :type numClass: int
    :param itemsList: list of items
    :type itemsList: list[Item]
    :return: The best bag that has all the classes.
    """
    bags = []
    for combination in itertools.product([0, 1], repeat=len(itemsList)):
        bag = Bag([], 0, 0, [0] * numClass)
        for i, selected in enumerate(combination):
            if selected:
                item = itemsList[i]
                if bag.weight + item.weight <= weightBag:
                    bag.add_item(item)
                else:
                    break
        if bag.weight <= weightBag:
            bags.append(bag)

    best_bag = Bag([], 0, 0, [0] * numClass)
    for bag in bags:
        if best_bag is None or bag.value > best_bag.value:
            if all(c > 0 for c in bag.ClassList):
                best_bag = bag
                
    return best_bag
  

def main():
#Index of file input
    n = 10
# It's a way to measure the time and memory used by the program.
    startTime = time.time()
    process = psutil.Process()

    weightBag = float
    numClass = 0
    weightBag, numClass, ItemList = readInputFile(n)
    print(f'FILE INPUT_{n}.txt')
    print("LOADING...")
    solution = bruteForce(weightBag, numClass, ItemList)
    writeOutputFile(n, solution, ItemList)

# It's a way to measure the time and memory used by the program.
    endTime = time.time()
    print("Time run is: " + str(endTime - startTime) + " seconds")
    print("Memory need: " + str(process.memory_info().rss / 1024 / 1024) + " MB")

main()