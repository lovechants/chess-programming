import numpy as np 
from model import * 
def fix_gene(genome):
    expected_len = genome['num_layers']
    genome['filters'] = genome['filters'][:expected_len]
    genome['activations'] = genome['activations'][:expected_len]
    return genome



def init_population(size):
    population = []
    for _ in range(size):
        num_layers = np.random.randint(1,5)
        genome = {
                'num_layers': num_layers,
                'filters': [np.random.choice([32,64,128]) for _ in range(num_layers)],
                'learning_rate': np.random.uniform(0.0001, 0.001),
                'activations': [np.random.choice(['relu', 'sigmoid', 'tanh']) for _ in range(num_layers)]
        }
        proper_genome = fix_gene(genome)
        population.append(proper_genome)
    return population

def eval_fitness(child, data):
    model = create_model(child)
    if model:
        fitness = eval_network(model, data)
        return fitness    
    else: return float('-inf')

def select(population, fitnesses, tourney_size=5): #tournament selection
    selected = []
    population_size = len(population)
    for _ in range(population_size):
        index = np.random.choice(range(population_size), tourney_size, replace=False)
        tournament = [(population[i], fitnesses[i]) for i in index]
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(fix_gene(winner))
    return selected

def crossover(p1, p2):
    if np.random.rand() > .7:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    keys = list(c1.keys()) 
    intersection = np.random.randint(1, len(p1))
    for key in keys[intersection:]:
        c1[key], c2[key] =  c2[key], c1[key]
    return fix_gene(c1), fix_gene(c2)

def mutate(child, mutation_rate=.1):
    for key in child.keys():
        if np.random.rand() < mutation_rate:
            if key == 'num_layers': 
                child[key] = np.random.randint(1,5)
            elif key == 'learning_rate':
                change_factor = 1.0 + np.random.choice([-1.0,1.0]) * .1
                new_rate = child[key] * change_factor
                child[key] = np.clip(new_rate, 0.0001, 0.01)
    return fix_gene(child) 

def genetic_algo(data, population, population_size=50, generations=100, tournament_size=5, mut_rate=.1):
    for generation in range(generations):
        fits = [eval_fitness(child, data) for child in population]
        if not all(isinstance(fit, (int, float)) for fit in fits):
            print("Non-Numeric Fits")
        selected = select(population, fits, tournament_size)
        if not selected:
            print("No Genomes selected, check logic")
        next_gen = []
        while len(next_gen) < population_size:
            parent1, parent2 = np.random.choice(selected, 2, replace=False)
            child1, child2 = crossover(parent1, parent2)
            child1 = fix_gene(child1)
            child2 = fix_gene(child2)
            next_gen.append(child1)
            if len(next_gen) < population_size:
                next_gen.append(child2)
        population = [mutate(fix_gene(child), mut_rate) for child in next_gen]
        best_fit = max(fits)
        print(f"Generation {generation}: Best Fit {best_fit}")
    final_fit = [eval_fitness(child, data) for child in population]
    if not all(isinstance(fit, (int, float)) for fit in final_fit):
            print("Non-Numeric Final Fits")
    best_index = np.argmax([eval_fitness(child, data) for child in population])
    best_genome = population[best_index]
    print(f"Selected best genome with index {best_index} and fitness {final_fit[best_index]}")
    return best_genome

