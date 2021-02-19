import numpy
import sqlite3
import pandas as pd
import xlrd
import random

"""
Reference:
1. https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
"""

### START OF GA MODULE ###
def cal_plant(data):
    leng = len(str(data))
    if leng == 1:
      return "PLANT0" + str(data)
    elif leng == 2:
      return "PLANT" + str(data)

    return "PLANT0" + str(data)

def cal_port(data):
    leng = len(str(data))
    if leng == 1:
      return "PORT0" + str(data)
    elif leng == 2:
      return "PORT" + str(data)

    return "PORT0" + str(data)

def cal_service(data):
    switcher = {
        1: "DTD",
        2: "DTP",
        3: "CRF"
    }
    return switcher.get(data, "Invalid Data")

def cal_mode(data):
    switcher = {
        1: "AIR",
        2: "GROUND"
    }
    return switcher.get(data, "Invalid Data")

def cal_random_value(value):
    # pop_data = numpy.random.randint([1, 1, 1, 1, 0, 1],[20, 12, 3, 4, 100000, 11], size=6)
    if value == 0:
        return numpy.random.randint(1,20)
    
    if value == 1:
        return numpy.random.randint(1,12)

    if value == 2:
        return numpy.random.randint(1,3)
    
    if value == 3:
        return numpy.random.randint(1,4)

    if value == 4:
        return numpy.random.randint(0,100000)

    if value == 5:
        return numpy.random.randint(1,11)

    return 0

def cal_pop_fitness(pop, db_conn, c):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    # WC
    arr = numpy.empty((0,7), int)

    for pop_data in pop:

      # pop_data[0] = "Plant"
      # pop_data[1] = "PORT"
      # pop_data[2] = "Mode "
      # pop_data[3] = "Service_Level "
      # pop_data[4] = "Weight "
      # pop_data[5] = "Carrier"

      while True:
        sqldtra = '''SELECT count(*) as t FROM pp where plant_code = ? and port = ?'''
        ca = c.execute(sqldtra, (cal_plant(pop_data[0]),cal_port(pop_data[1])))
        (number_of_rows,)=ca.fetchone()

        sqldata = '''SELECT minimum_cost, rate, svc_cd FROM fr where orig_port_cd = ? and svc_cd = ? and carrier = ? and mode_dsc like ? order by rate asc limit 1'''
        cu = c.execute(sqldata, (cal_port(pop_data[1]), cal_service(pop_data[3]), "V444_"+str(pop_data[5]), "%"+ cal_mode(pop_data[2]) +"%"))
        data = cu.fetchall()

        if number_of_rows == 0 or len(data) == 0:
          pop_data = numpy.random.randint([1, 1, 1, 1, 0, 1],[20, 12, 3, 4, 100000, 11], size=6)
        else:
          break
          
      for row in data:
        minimumCost = row[0]
        rate = row[1]
        svc = row[2] #dtd,dtp,crf

      tc = round(rate, 2) *  pop_data[4] #WEIGHT 
    
      if tc < minimumCost:
          tc = minimumCost
      elif cal_service(pop_data[3]) == "CRF":
          tc = 0

      #wh warehouse
      wcf = '''SELECT CostUnit FROM wc where Wh = ?'''
      ca = c.execute(wcf, (cal_plant(pop_data[0]),))
      (CostUnit,)=ca.fetchone()
      wh = round(CostUnit, 2) * pop_data[4] #weight

      currentCost = int(wh) + int(tc)

      #  sort
      z = numpy.array(currentCost)
      d = numpy.array(pop_data)

      j = numpy.append(d, numpy.array([z]), axis=0)
      arr = numpy.append(arr, numpy.array([j]), axis=0)

    # Sorting By CurrentCost
    arr = arr[arr[:,6].argsort()]

    # Remove Temp Last Index currectCost
    data1 = [item[:6] for item in arr]

    data2 = [item[6:] for item in arr]

    return numpy.array(data1), numpy.array(data2)

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size, dtype=int)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    i = 0
    for k in range(int(offspring_size[0]/2)):
        # Index of the first parent to mate.
        parent1_idx = i
        # Index of the second parent to mate.
        parent2_idx = i+1
        
        #Individu 1 = 2 5 | 1 2 | 200 5
        #Individu 2 = 3 8 | 2 1 | 500 3

        #Offspring 1 = 3 8 | 1 2 | 500 3
        #Offspring 2 = 2 5 | 2 1 | 200 5

        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[i, 0:2] = parents[parent2_idx, 0:2]
        offspring[i, 2:4] = parents[parent1_idx, 2:4]
        offspring[i, 4:6] = parents[parent2_idx, 4:6]
        
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[i+1, 0:2] = parents[parent1_idx, 0:2]
        offspring[i+1, 2:4] = parents[parent2_idx, 2:4]
        offspring[i+1, 4:6] = parents[parent1_idx, 4:6]
        i = i+2

    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    # The random value to be added to the gene. base on the random index
    # idx = numpy.random.randint(0,6)
    # change = numpy.random.randint(0,6)
    # print("idx and change_idx", idx +" and "+ change)
    ranged = 2
    idx = random.sample(range(0, 6), ranged)

    for x in range(ranged):
      random_value = cal_random_value(idx[x])
      offspring_crossover[idx[x]] = random_value
    
    return offspring_crossover

### END OF GA MODULE

sol_per_pop = 20
num_parents_mating = 6
num_weights = 6
# Defining the population size.
pop_size = (sol_per_pop, num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
best_outputs = []
num_generations = 100
prob_percentage = 0.4

db_conn = sqlite3.connect("ga.db")
c = db_conn.cursor()

whc = pd.read_excel('/content/dataset.xlsx', sheet_name='WhCosts', header=0)
whc.to_sql('wc', db_conn, if_exists='replace', index=False)

tc = pd.read_excel('/content/dataset.xlsx', sheet_name='FreightRates', header=0)
tc.to_sql('fr', db_conn, if_exists='replace', index=False)

plantport = pd.read_excel('/content/dataset.xlsx', sheet_name='PlantPorts', header=0)
plantport.to_sql('pp', db_conn, if_exists='replace', index=False)

new_population = numpy.random.randint([1, 1, 1, 1, 0, 1],[20, 12, 3, 4, 100000, 11], size=pop_size)
prob = [1 if random.random() < prob_percentage else 0 for _ in range(num_generations)]
for generation in range(num_generations):

    print("generation", generation)

    if generation > 0:
       pop_size = (sol_per_pop-num_parents_mating, 6)
       new_population = numpy.random.randint([1, 1, 1, 1, 0, 1],[20, 12, 3, 4, 100000, 11], size=pop_size)
       new_population = numpy.append(new_population, new_pop, axis=0)

    # Measuring the fitness of each chromosome in the population.
    fitness, data = cal_pop_fitness(new_population, db_conn, c)
    # print("Fitness", data[0])
    print("Fitness Score ", data[0])
    # print(fitness)

    best_outputs.append(data[0])

    # The best result in the current iteration.
    print("Best result : ", fitness[0])
    
    # Selecting the best parents in the population for mating.
    # parents = select_mating_pool(new_population, fitness, num_parents_mating)
    parents = fitness[:num_parents_mating]

    print("parents", parents)
    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(num_parents_mating, num_weights))
    print("offspring_crossover", offspring_crossover)

    # Adding some variations to the offspring using mutation.
    new_pop = offspring_crossover

    if prob[generation] == 1:
      new_pop = mutation(new_pop)#, num_mutations=2)
      print("mutation_population", new_pop)
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(new_population, db_conn, c)
# Then return the index of that solution corresponding to the best fitness.
# best_match_idx = numpy.where(fitness == numpy.min(fitness))

# print("best_match_idx", best_match_idx)
# print("Best solution : ", new_population[best_match_idx, :])
# print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
