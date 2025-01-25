import numpy as np
import time
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.operator import BitFlipMutation
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem import OneMax

def local_search_run(number_of_bits, max_evaluations, mutation_probability):
    # Crée une instance du problème
    problem = OneMax(number_of_bits=number_of_bits)

    # Définit la probabilité de mutation par défaut si elle n'est pas spécifiée
    mutation_probability = mutation_probability / problem.total_number_of_bits()

    # Configure l'algorithme de recherche locale
    algorithm = LocalSearch(
        problem=problem,
        mutation=BitFlipMutation(probability=mutation_probability),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(100))

    # Mesure le temps de calcul
    start_time = time.time()
    algorithm.run()
    end_time = time.time()

    result = algorithm.solutions[0]

    # Retourne les résultats sous forme de dictionnaire
    return {
        "fitness": result.objectives[0],
        "solution": result.get_binary_string(),
        "computing_time": end_time - start_time,
    }

def experiment_runs(runs, number_of_bits, max_evaluations, mutation_probability):
    results = []

    for run in range(runs):
        result = local_search_run(
            number_of_bits=number_of_bits,
            max_evaluations=max_evaluations,
            mutation_probability=mutation_probability,
            verbose=False,
        )
        results.append(result)

    # Analyse statistique des résultats
    fitness_values = [res["fitness"] for res in results]
    computing_times = [res["computing_time"] for res in results]

    stats = {
        "fitness_mean": np.mean(fitness_values),
        "fitness_median": np.median(fitness_values),
        "fitness_std": np.std(fitness_values),
        "time_mean": np.mean(computing_times),
        "time_median": np.median(computing_times),
        "time_std": np.std(computing_times),
        "all_results": results,
    }

    return stats

if __name__ == "__main__":
    # Paramètres pour l'expérience
    number_of_bits = 512
    max_evaluations = 10000
    mutation_probability = 1.0 / number_of_bits
    runs = 20

    # Lancer les expériences
    stats = experiment_runs(
        runs=runs,
        number_of_bits=number_of_bits,
        max_evaluations=max_evaluations,
        mutation_probability=mutation_probability,
    )

    # Afficher la synthèse des résultats
    print("\nSynthèse des résultats (20 runs):")
    print(f"Fitness moyenne : {stats['fitness_mean']:.2f}")
    print(f"Fitness médiane : {stats['fitness_median']:.2f}")
    print(f"Écart-type de la fitness : {stats['fitness_std']:.2f}")
    print(f"Temps de calcul moyen : {stats['time_mean']:.2f} s")
    print(f"Temps de calcul médian : {stats['time_median']:.2f} s")
    print(f"Écart-type du temps de calcul : {stats['time_std']:.2f} s")