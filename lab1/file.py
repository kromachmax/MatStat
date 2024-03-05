import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
np.random.seed(42)

normal = np.random.standard_normal(900)
cauchy = np.random.standard_cauchy(201)
student = np.random.standard_t(df=3, size=1001)
poisson = np.random.poisson(10, size=1001)
uniform = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 1001)

array_of_powers = [10, 50, 1000]

array_of_intervals_normal = [7, 20, 30]
array_of_intervals_cauchy = [5, 12, 30]
array_of_intervals_student = [7, 20, 30]
array_of_intervals_poisson = [3, 10, 30]
array_of_intervals_uniform = [3, 10, 30]

arr_intervals = [array_of_intervals_normal, array_of_intervals_cauchy, array_of_intervals_student, array_of_intervals_poisson, array_of_intervals_uniform]
array_of_distributions = [normal, cauchy, student, poisson, uniform]
names = ["normal", "cauchy", "student", "puasson", "uniform"]

for distribution, name, intervals in zip(array_of_distributions, names, arr_intervals):
    for power, interval in zip(array_of_powers, intervals):
        # Создаем гистограмму с заполненными столбцами
        plt.figure(figsize=(8, 6))
        bin_values, bin_edges, _ = plt.hist(distribution[:power], bins=interval, density=True, alpha=0.7, edgecolor='black')
        # Замена пустых столбцов на столбцы с высотой 1
        for i in range(len(bin_values)):
            if bin_values[i] == 0:
                bin_values[i] = (bin_values[i - 1] + bin_values[i + 1]) / 2
        # Рисуем заполненные столбцы
        plt.bar(bin_edges[:-1], bin_values, width=np.diff(bin_edges), align='edge', alpha=0.7, edgecolor='black', color='cyan')
        # Добавляем график аппроксимации
        sns.kdeplot(distribution[:power], color='red')
        plt.title(f'{name}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.savefig(f'{name}, size {power}')