import math
import random
import matplotlib.pyplot as plt
import csv


# Given Problem
def fitness_eqn(x_dict):
    try:
        with open("equation.txt") as eqn_file:
            expression = eqn_file.readline()
    except:
        print("\n\tEquation file is not present.")
        exit(1)
    return eval(expression, x_dict)


# Function to calculate fitness values
def fitness_values(Min_or_Max, x):
    fit_val = []
    for i in x:
        x_dict = {}
        for j in range(len(i)):
            variable = f'x{j + 1}'
            x_dict[variable] = i[j]
        if Min_or_Max == 1:
            # fit_val.append(-1 * fitness_eqn(x_dict))
            fit_val.append(1/(1+fitness_eqn(x_dict)))
        elif Min_or_Max == 2:
            fit_val.append(fitness_eqn(x_dict))
    return fit_val


# Function to display optimal solution
def disp_fun_val(x):
    x_dict = {}
    for i in range(len(x)):
        variable = f'x{i + 1}'
        x_dict[variable] = x[i]
    return fitness_eqn(x_dict)


# Function to Generate Population
def generate_population(pop_size, string_len):
    population = []
    for i in range(pop_size):
        chromosome = []
        for j in range(string_len):
            chromosome.append(random.randint(0, 1))
        population.append(chromosome)
    return population


# Function to Decode
def decode(population, n):
    values = []
    for chromosome in population:
        lst = [chromosome[i:i+n] for i in range(0, len(chromosome), n)]
        each_variable = []
        for i in lst:
            i.reverse()
            sum = 0
            for j in range(len(i)):
                sum += i[j] * math.pow(2, j)
            each_variable.append(sum)
        values.append(each_variable)
    return values


# Function to calculate X-values
def x_values(x, population):
    pop_size = len(population)
    string_len = len(population[0])
    number_of_variables = len(x)
    x_calculated = []
    each_variable = int(string_len / number_of_variables)
    decoded_values = decode(population, each_variable)
    for i in range(pop_size):
        row_val = []
        for j in range(number_of_variables):
            x_calc = x[j][0] + (((x[j][1]-x[j][0])/(math.pow(2,each_variable)-1)) * decoded_values[i][j])
            row_val.append(x_calc)
        x_calculated.append(row_val)
    return x_calculated


# Function for two point crossover
def two_point_crossover(mating_pool, crossover_prob):
    pop_size = len(mating_pool)
    chosen = []
    check = []
    parents = mating_pool.copy()
    for i in range(pop_size // 2):
        items = []
        for j in range(2):
            x = random.choice([k for k in range(pop_size) if k not in check])
            items.append(x)
            check.append(x)
        chosen.append(items)
    for i in chosen:
        mating_prob = random.uniform(0, 1)
        if mating_prob <= crossover_prob:
            crossover_sites = []
            for j in range(2):  # Two point crossover
                crossover_sites.append(random.choice([n for n in range(0, len_of_string - 1) if n not in crossover_sites]))
            parent_1 = parents[i[0]]
            parent_2 = parents[i[1]]
            child_1 = []
            child_2 = []
            for k in range(len_of_string):
                if min(crossover_sites) <= k <= max(crossover_sites):
                    child_1.append(parent_2[k])
                    child_2.append(parent_1[k])
                else:
                    child_1.append(parent_1[k])
                    child_2.append(parent_2[k])
            parents[i[0]] = child_1
            parents[i[1]] = child_2
    return parents


# Function for mutation
def mutation(population, probability):
    pop_size = len(population)
    string_len = len(population[0])
    mutated = population.copy()
    for i in range(pop_size):
        for k in range(string_len):
            r = random.uniform(0,1)
            if r <= probability:
                if mutated[i][k] == 0:
                    mutated[i][k] = 1
                else:
                    mutated[i][k] = 0
    return mutated


# Function to display optimal solution and their corresponding variables
def display(fitness_string, x_val_string):
    string = "\n\nVariables:\n"
    number_of_variables = len(x_val_string[0])
    index_of_max = fitness_string.index(max(fitness_string))
    x = []
    for i in range(number_of_variables):
        string += f"\tx{i+1} = {x_val_string[index_of_max][i]}"
        x.append(x_val_string[index_of_max][i])
    string += f"\n\nOptimal solution is {disp_fun_val(x)}"
    return string


# Inputs
print('\nEnter the objective function in Equation.txt file:')
Min_or_Max = int(input('Choose option:\n1 - Minimisation.\n2 - Maximisation.\n'))

# Number of variables and Boundary conditions
with open("equation.txt") as eqn_file:
    file_content = eqn_file.readlines()

file_content = [i for i in file_content if i != '\n']

number_of_variables = len(file_content) - 1
x = []
for i in range(number_of_variables):
    bound_cond = file_content[i+1].split()
    x.append(list(map(float, bound_cond)))

size_of_population = int(input("\nEnter Size of Population: "))
crossover_prob = float(input("Enter Cross-over Probability: "))
mutation_prob = float(input("Enter Mutation Probability: "))
max_gen = 100

# Length of string
epsilon = 0.0001  # Accuracy
len_of_string = 0
for i in range(number_of_variables):
    len_of_string += math.ceil(math.log2((x[i][1] - x[i][0])/epsilon))

# Initial population
population = generate_population(size_of_population, len_of_string)

gen = 0
err = 1
avg_fitness = []
generation = []
max_fitness = []
min_fitness = []
test_cnt = 0
fitness = 0
x_val = 0
x1_optimal = []
x2_optimal = []

while(gen <= max_gen) and (err > 1e-10):

    x1_values = []
    x2_values = []

    # Calculate X_values
    x_val = x_values(x, population)
    # Fitness values
    fitness = fitness_values(Min_or_Max, x_val)
    fitness_sum = sum(fitness)
    fitness_avg = fitness_sum/size_of_population

    # Probability calculation
    probability = []
    for i in range(size_of_population):
        probability.append(fitness[i]/fitness_sum)

    # Creating mating pool using Roulette Wheel selection
    sorted_probability = sorted(probability)
    mating_pool = []
    for i in range(size_of_population):
        prob_value = 0
        p = random.uniform(0, 1)
        for j in range(size_of_population):
            prob_value += sorted_probability[j]
            if p <= prob_value:
                z = probability.index(sorted_probability[j])
                mating_pool.append(population[z])
                break

    # Two point crossover
    child_population = two_point_crossover(mating_pool, crossover_prob)

    # Mutation
    child_population = mutation(child_population, mutation_prob)

    population = child_population.copy()

    if gen > 10:
        err = abs(avg_fitness[gen-1] - avg_fitness[gen-11])

    for i in range(size_of_population):
        x1_values.append(x_val[i][0])
        x2_values.append(x_val[i][1])

    x1_optimal.append(sum(x1_values)/size_of_population)
    x2_optimal.append(sum(x2_values)/size_of_population)

    generation.append(gen)
    avg_fitness.append(fitness_avg)
    max_fitness.append(max(fitness))
    min_fitness.append(min(fitness))

    gen += 1

print(display(fitness, x_val))

# Various Plots
plt.plot(generation, avg_fitness)
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.title('Average Fitness vs Generations')
plt.savefig('Average Fitness vs Generations.png')
plt.show()
plt.plot(generation, max_fitness, label='Maximum Fitness')
plt.plot(generation, min_fitness, label='Minimum Fitness')
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title('Fitness vs Generations')
plt.legend()
plt.savefig('Fitness vs Generations.png')
plt.show()
plt.plot(generation, x1_optimal, label='X1 Values')
plt.plot(generation, x2_optimal, label='X2 Values')
plt.xlabel('Generations')
plt.ylabel('values of variables')
plt.title('Variable values vs Generations')
plt.legend()
plt.savefig('Variable values vs Generations.png')
plt.show()