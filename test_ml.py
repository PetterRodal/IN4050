import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
np.random.seed(57)
#Map of Europe
europe_map = plt.imread('map.png')

#Lists of city coordinates
city_coords = {
    "Barcelona": [2.154007, 41.390205], "Belgrade": [20.46, 44.79], "Berlin": [13.40, 52.52], 
    "Brussels": [4.35, 50.85], "Bucharest": [26.10, 44.44], "Budapest": [19.04, 47.50],
    "Copenhagen": [12.57, 55.68], "Dublin": [-6.27, 53.35], "Hamburg": [9.99, 53.55], 
    "Istanbul": [28.98, 41.02], "Kyiv": [30.52, 50.45], "London": [-0.12, 51.51], 
    "Madrid": [-3.70, 40.42], "Milan": [9.19, 45.46], "Moscow": [37.62, 55.75],
    "Munich": [11.58, 48.14], "Paris": [2.35, 48.86], "Prague": [14.42, 50.07],
    "Rome": [12.50, 41.90], "Saint Petersburg": [30.31, 59.94], "Sofia": [23.32, 42.70],
    "Stockholm": [18.06, 60.33], "Vienna": [16.36, 48.21], "Warsaw": [21.02, 52.24]}

#Helper code for plotting plans
#First, visualizing the cities.
import csv
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

# Map (long, lat) to (x, y) for plotting
for city, location in city_coords.items():
    x, y = (location[0], location[1])
    plt.plot(x, y, 'ok', markersize=5)
    plt.text(x, y, city, fontsize=12)


#A method you can use to plot your plan on the map.
def plot_plan(city_order):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

    # Map (long, lat) to (x, y) for plotting
    for index in range(len(city_order) - 1):
        current_city_coords = city_coords[city_order[index]]
        next_city_coords = city_coords[city_order[index+1]]
        x, y = current_city_coords[0], current_city_coords[1]
        #Plotting a line to the next city
        next_x, next_y = next_city_coords[0], next_city_coords[1]
        plt.plot([x, next_x], [y, next_y])

        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x, y, index, fontsize=12)
    #Finally, plotting from last to first city
    first_city_coords = city_coords[city_order[0]]
    first_x, first_y = first_city_coords[0], first_city_coords[1]
    plt.plot([next_x, first_x], [next_y, first_y])
    #Plotting a marker and index for the final city
    plt.plot(next_x, next_y, 'ok', markersize=5)
    plt.text(next_x, next_y, index+1, fontsize=12)
    plt.show()

    #Example usage of the plotting-method.
plan = list(city_coords.keys()) # Gives us the cities in alphabetic order
print(plan)
plot_plan(plan)

def calculate_total_distance(order, distances): #Function that calculates the total distance based on city order. NB!  This function is auto generated using CHATgpt.
   
   total_distance = 0
   for i in range(len(order) - 1): 
      total_distance += distances[order[i]][order[i + 1]]
   total_distance += distances[order[-1]][order[0]]
   return total_distance

def data_to_float(length):  #Function that gets the data from the csv file and converts it to float

   with open('european_cities.csv', 'r') as file:
      csv_reader = csv.reader(file, delimiter = ';')

      #Skips first row since that row is only city names
      header = next(csv_reader, None)


      data = [next(csv_reader)[:len(cities)] for _ in range(len(cities))] #Gets the rows and columns from csv file needed 

   distances = np.array(data, dtype=str)
   
   #Converts the data from string to float
   distances = [
    [float(element) for element in row] 
    for row in distances
   ]

   return distances

import random

def inital_tour(cities): #Starting tour is random for each run
    return random.sample(range(cities), cities)


def calculate_total_distance(tour, distances):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distances[tour[i]][tour[i + 1]]
    total_distance += distances[tour[-1]][tour[0]]
    return total_distance

# Implement the algorithm here

#Reprsentation: Candidate solutions is representated as a permutation of the cities. Each gene in the chromosome represent a city, and the order of the genes corresponds to the order in whcich the cities are visited. 

import random

import time

start2 = time.time()

cities = plan[:24]

#Will be using swap mutation

def swap_mutation(offspring):

    index1, index2 = random.sample(range(len(offspring)), 2) #Pick to random indices (genes) in the offspring and swaps them

    offspring[index1], offspring[index2] = offspring[index2], offspring[index1]

    return offspring


def order_crossover(parent1, parent2):
    offspring1 = [None] * len(parent1) 
    offspring2 = [None] * len(parent2)

    start, end = sorted(random.sample(range(len(parent1)),2)) #Pick two random indicies (genes)

    segment1 = parent1[start:end] #Gets the genes between the indices

    offspring1[start:end] = segment1 #Adds those to the offspring
 
    P_missing = []
    P_missing.extend(value for value in parent1 if value not in segment1) #Find the remaining genes that needs to be transferred in order in which thay appear in parent2
    P_missing.sort(key=lambda x: parent2.index(x))

    testing = []

    offspring1[0:start] = P_missing[0:start] #Adds the reamining genes to the offspring
    testing.extend(value for value in P_missing if value not in offspring1) 
    offspring1[end:len(parent1)] = testing

    #Creating the second offspring with parents role reversed 

    segment2 = parent2[start:end]

    offspring2[start:end] = segment2

    P_missing2 = []
    P_missing2.extend(value for value in parent2 if value not in segment2)
    P_missing2.sort(key=lambda x: parent1.index(x))

    testing2 = []

    offspring2[0:start] = P_missing2[0:start]
    testing2.extend(value for value in P_missing2 if value not in offspring2)
    offspring2[end:len(parent2)] = testing2


    return offspring1, offspring2


def crossover(parents): #
    offspring_list = []
    while len(parents) >= 2:
        parent1, parent2 = random.sample(parents, 2)
        parents.remove(parent1)
        parents.remove(parent2)

        offspring1, offspring2 = order_crossover(parent1, parent2)

        offspring_list.extend([offspring1, offspring2])
    
    return offspring_list



#Ranking selection
def parent_selection(population, selection_pressure = 1.5):
    population_size = len(population)

    
    fitness_scores = []
    for pop in population:
        
        dic = data_to_float(pop)
        
        fitness = calculate_total_distance(pop, dic)
        fitness_scores.append(fitness)



    ranked_population = sorted(population, key=lambda x: calculate_total_distance(x, data_to_float(x))) #Sortes the population based on fitness
    


    prob_list = [] #List of the probabilties of being selected as a parent, which is propotional to the rankings
    i = 0
    for i in range(len(ranked_population)):
        probality = ((2 - selection_pressure) / population_size) + (2 * i * (selection_pressure - 1)) / (population_size * (population_size - 1))
        prob_list.append(probality)
        i += 1
    
    selected_parents = []
    
    #Using roulettte wheel to select the parents

    #NB this implmentation is auto generated using CHATgpt

    """"

    while True:
        rand_num1 = random.random()
        try:
            selected_parent_index1 = next(i for i, prob in enumerate(prob_list) if prob >= rand_num1)
            break
        except StopIteration:
            pass

    parent1 = ranked_population[selected_parent_index1]

    selected_parents.append(parent1)

    # Select the second parent and ensuring it's different from the first parent
    while True:
        rand_num2 = random.random()
        selected_parent_index2 = next((i for i, prob in enumerate(prob_list) if prob >= rand_num2), None)
        if selected_parent_index2 is not None and selected_parent_index2 != selected_parent_index1:
            parent2 = ranked_population[selected_parent_index2]
            break
    
    selected_parents.append(parent2)

    """

    selected_indices = set()
    max_parents = min(population_size // 2, len(ranked_population) // 2) 

    while len(selected_parents) < max_parents: #Half of the population is selected as parents
        rand_num = random.random()
        selected_parent_index = next((i for i, prob in enumerate(prob_list) if prob >= rand_num and i not in selected_indices), None)
        if selected_parent_index is not None:
            selected_indices.add(selected_parent_index)
            selected_parents.append(ranked_population[selected_parent_index])

    return selected_parents

    
def inital_tour(cities):
    return random.sample(range(cities), cities)



def survivor_selection(population ,offsprings):
    survivors = []

    for ele in population:
        survivors.append(ele)

    for child in offsprings:
        survivors.append(child)
    
    survivors_test = sorted(survivors, key=lambda x: calculate_total_distance(x, data_to_float(x))) #Picking the fitess in the new population
    
    survivor_selected = []
    survivor_selected.extend(survivors_test[:len(survivors) - 2]) #Ensuring the population dosen't increase or decrease

    #print(survivor_selected)

    survivors.clear()

    return survivor_selected

        


def tsp_genetic_algorithm(cities, population_size, generations):
    num_cities = len(cities)
    population = []

    for _ in range(population_size):
        population.append(inital_tour(num_cities))

    
    y_axis = []

    mutation_probability = 0.05

    for _ in range(generations):
        #population = sorted(population, key=lambda x: calculate_total_distance(x, data_to_float(x)))
        parents = parent_selection(population)
        offsprings = crossover(parents)
       
        for child in offsprings:
            if random.random() < mutation_probability:
                swap_mutation(child)
    
        population = survivor_selection(population, offsprings)
        best_tours = min(population, key=lambda x: calculate_total_distance(x, data_to_float(x)))
        y_axis.append(calculate_total_distance(best_tours, data_to_float(best_tours)))




    best_tour = min(population, key=lambda x: calculate_total_distance(x, data_to_float(x)))
   
    return best_tour, calculate_total_distance(best_tour, data_to_float(best_tour)), y_axis



# Generating x-axis values from 1 to 100 for plotting
x_values = list(range(1, 101))



def tsp_multiple_runs(diff_populations, iterations):

    best_tours = []
    
    avarage_best_tour = np.empty((0, iterations), dtype=float) 
    for _ in range(20):
        besttour, disctance, potter = tsp_genetic_algorithm(cities, diff_populations, iterations)
        avarage_best_tour = np.vstack([avarage_best_tour, potter]) #Stacks all the best tours in a array, so its easier to find the avarage best tour for each generation
        best_tours.append(besttour)
        
    average_over_gen = np.mean(avarage_best_tour, axis=0) #Gets the avarage for each column (=Generation) in array.
    average_list = average_over_gen.tolist()

    abs_best_tour = min(best_tours, key=lambda x: calculate_total_distance(x, data_to_float(x))) 

    best_fit_ind = avarage_best_tour[:, iterations - 1] #Adds the last individuals in the last generation to a list

    best_distances_tsp = np.min(best_fit_ind)
    worst_distances_tsp = np.max(best_fit_ind)
    mean_dis = np.mean(best_fit_ind)
    std_dev = np.std(best_fit_ind)

    return average_list, best_distances_tsp, worst_distances_tsp, mean_dis, std_dev, abs_best_tour


population_sizes = [6]

plotting_cities = []

for pop in population_sizes:
    test1, be, wo, mean, std, best = tsp_multiple_runs(pop, 100)
    print(f"For population size of {pop}, the best fit individuals in last generation: ")
    print("Best Distance:", round(be, 2))
    print("Worst Distance:", round(wo,2))
    print("Mean distance:", round((mean), 2))
    print("Standard Deviation:", round(std, 2))
    print("")
    plt.plot(x_values, test1, label=f'Population size: {pop}', marker='')
    best_cities =  [cities[i] for i in best]
    plotting_cities.append(best_cities)

    



# Creating the plot

plt.title('Plot of the average fitness of the best tour over genereations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()

plt.show()


for tours in plotting_cities:
    plot_plan(tours)

end_time2 = time.time()
total_time2 = end_time2 - start2
print(f"Total running time: {round(total_time2, 3)} seconds")
