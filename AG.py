#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygad
import numpy as np
from benchmark import *
from cell_lib import *

# In[23]:


def fitness_func(solution, solution_idx):
    dp      = solution[0]
    minDist = int(solution[1])
    param1  = int(solution[2])
    param2  = int(solution[3]) 
    

    
    path = "./samples/normal/blood_smear_1.JPG"
    
    images, stats = hough_circles_method(path,dp,minDist,param1,param2)
    errors = 40
    
    errors, img, manual = compare_markers("./samples/normal/blood_smear_1_count.JPG", images[0])
    
    return 1.0 - errors/100
    
    


# In[24]:


fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = 4

gene_space = [{'low': 0.1, 'high': 5.0}, #dp
              {'low':1, 'high':30},#minDist
              {'low':1, 'high':100},#param1
              {'low':1, 'high':100}]#param2
              

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


# In[25]:


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)
ga_instance.run()


# In[27]:


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))


# In[30]:



a = np.array([-1.26154444, -1.57882543,  0.13756688,  0.4352141,  -3.34907296, -1.35826638]) * np.array([4.0,-2.0,3.5,5.0,-11.0,-4.7])


# In[32]:


np.sum(a)

