import numpy as np
import statistics
import time
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem import OneMax
from jmetal.util.termination_criterion import StoppingByEvaluations

# Fonction pour exécuter l'algorithme avec des paramètres modifiables
def genetic_algorithm_run(number_of_bits, population_size, offspring_population_size, max_evaluations, mutation_prob, crossover_prob):
    # Problème
    problem = OneMax(number_of_bits=number_of_bits)

    # Opérateurs
    mutation = BitFlipMutation(probability=mutation_prob/number_of_bits)
    crossover = SPXCrossover(probability=crossover_prob)

    # Algorithme
    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=population_size,
        offspring_population_size=offspring_population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    # Exécution et mesure du temps
    start_time = time.time()
    algorithm.run()
    end_time = time.time()

    # Résultats
    result = algorithm.solutions[0]

    return {
        "fitness": result.objectives[0],
        "solution": result.get_binary_string(),
        "computing_time": end_time - start_time
    }

# Fonction pour exécuter plusieurs runs et calculer les statistiques
def experiment_runs(runs, number_of_bits, population_size, offspring_population_size, max_evaluations, mutation_prob, crossover_prob):
    results = []

    for _ in range(runs):
        result = genetic_algorithm_run(
            number_of_bits=number_of_bits,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            max_evaluations=max_evaluations,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
        )
        results.append(result)

    # Analyse des résultats
    fitness_values = [res["fitness"] for res in results]
    computing_times = [res["computing_time"] for res in results]

    stats = {
        "fitness_mean": np.mean(fitness_values),
        "fitness_median": np.median(fitness_values),
        "fitness_std": np.std(fitness_values),
        "time_mean": np.mean(computing_times),
        "time_median": np.median(computing_times),
        "time_std": np.std(computing_times),
        "all_results": results
    }

    return stats

if __name__ == "__main__":
    # Paramètres pour les runs
    number_of_bits = 512
    population_size = 40
    offspring_population_size = 40
    max_evaluations = 20000
    mutation_prob = 1.0 / number_of_bits
    crossover_prob = 1.0
    runs = 20

    # Exécution des expériences
    stats = experiment_runs(
        runs=runs,
        number_of_bits=number_of_bits,
        population_size=population_size,
        offspring_population_size=offspring_population_size,
        max_evaluations=max_evaluations,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob
    )

    # Affichage des résultats synthétisés
    print("\nSynthèse des résultats après 20 runs :")
    print(f"Fitness moyenne : {stats['fitness_mean']:.2f}")
    print(f"Fitness médiane : {stats['fitness_median']:.2f}")
    print(f"Écart-type de la fitness : {stats['fitness_std']:.2f}")
    print(f"Temps de calcul moyen : {stats['time_mean']:.2f} s")
    print(f"Temps de calcul médian : {stats['time_median']:.2f} s")
    print(f"Écart-type du temps de calcul : {stats['time_std']:.2f} s")