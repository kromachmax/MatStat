import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def drawInfoNormal(array_of_dispersion, array_of_mathematical_expectation, powers):

        # Доверительные интервалы средневыборочного матожидания
        plt.figure(figsize=(10, 6))
        plt.hlines(y = 1, xmin = array_of_dispersion[0], xmax = array_of_dispersion[1], label = f'n = {powers[0]}')
        plt.plot([array_of_dispersion[0], array_of_dispersion[0]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_dispersion[1], array_of_dispersion[1]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.hlines(y = 1.1, xmin = array_of_dispersion[2], xmax = array_of_dispersion[3], label = f'n = {powers[1]}')
        plt.plot([array_of_dispersion[2], array_of_dispersion[2]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_dispersion[3], array_of_dispersion[3]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.title('Доверительный интервал дисперсии для нормального распределения')
        plt.savefig(f'{"intervalDispersion_normal"}')

        # Доверительные интервалы средневыборочного матожидания
        plt.figure(figsize=(10, 6))
        plt.hlines(y = 1, xmin = array_of_mathematical_expectation[0], xmax = array_of_mathematical_expectation[1], label = f'n = {powers[0]}')
        plt.plot([array_of_mathematical_expectation[0], array_of_mathematical_expectation[0]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_mathematical_expectation[1], array_of_mathematical_expectation[1]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.hlines(y = 1.1, xmin = array_of_mathematical_expectation[2], xmax = array_of_mathematical_expectation[3], label = f'n = {powers[1]}')
        plt.plot([array_of_mathematical_expectation[2], array_of_mathematical_expectation[2]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_mathematical_expectation[3], array_of_mathematical_expectation[3]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.title('Доверительный интервал матожидания для нормального распределения')
        plt.savefig(f'{"intervalMathematicalExpectation_normal"}')

def drawInfoPoisson(array_of_dispersion, array_of_mathematical_expectation, powers):

        # Доверительные интервалы средневыборочного матожидания
        plt.figure(figsize=(10, 6))
        plt.hlines(y = 1, xmin = array_of_dispersion[0], xmax = array_of_dispersion[1], label = f'n = {powers[0]}')
        plt.plot([array_of_dispersion[0], array_of_dispersion[0]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_dispersion[1], array_of_dispersion[1]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.hlines(y = 1.1, xmin = array_of_dispersion[2], xmax = array_of_dispersion[3], label = f'n = {powers[1]}')
        plt.plot([array_of_dispersion[2], array_of_dispersion[2]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_dispersion[3], array_of_dispersion[3]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.title('Доверительный интервал дисперсии для распределения Пуассона')
        plt.savefig(f'{"intervalDispersion_Poisson"}')

        # Доверительные интервалы средневыборочного матожидания
        plt.figure(figsize=(10, 6))
        plt.hlines(y = 1, xmin = array_of_mathematical_expectation[0], xmax = array_of_mathematical_expectation[1], label = f'n = {powers[0]}')
        plt.plot([array_of_mathematical_expectation[0], array_of_mathematical_expectation[0]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_mathematical_expectation[1], array_of_mathematical_expectation[1]], [1, 1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.hlines(y = 1.1, xmin = array_of_mathematical_expectation[2], xmax = array_of_mathematical_expectation[3], label = f'n = {powers[1]}')
        plt.plot([array_of_mathematical_expectation[2], array_of_mathematical_expectation[2]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Начальная точка
        plt.plot([array_of_mathematical_expectation[3], array_of_mathematical_expectation[3]], [1.1, 1.1], marker='o', markersize=10, color='purple', zorder=3)  # Конечная точка
        plt.legend()
        plt.title('Доверительный интервал матожидания для распределения Пуассона')
        plt.savefig(f'{"intervalMathematicalExpectation_Poisson"}')


def calculate_normal(quantiles, powers):
    print("normal_distribution:")
    bins = [10, 18]
    array_of_dispersion = []
    array_of_mathematical_expectation = []
    for power, quantile, bin in zip(powers, quantiles, bins):
        normal = np.random.normal(0, 1, power)

        average = np.mean(normal)

        dispersion = np.std(normal) 

        LeftMathematicalExpectation = average - dispersion * quantile[0] / math.sqrt(power - 1)

        RightMathematicalExpectation = average + dispersion * quantile[0] / math.sqrt(power - 1)

        LeftDispersion = dispersion * math.sqrt(power) / math.sqrt(quantile[1])

        RightDispersion = dispersion * math.sqrt(power) / math.sqrt(quantile[2])

        print(str(round(LeftMathematicalExpectation, 3)) + " < m < " + str(round(RightMathematicalExpectation, 3)))

        print(str(round(LeftDispersion, 3)) + " < s < " + str(round(RightDispersion, 3)))

        array_of_dispersion.append(LeftDispersion)
        array_of_dispersion.append(RightDispersion)
        array_of_mathematical_expectation.append(LeftMathematicalExpectation)
        array_of_mathematical_expectation.append(RightMathematicalExpectation)

        # Создаем гистограмму с заполненными столбцами
        plt.figure(figsize=(8, 6))
        bin_values, bin_edges, _ = plt.hist(normal, bins=bin, density=True, alpha=0.7, edgecolor='black')

        # Замена пустых столбцов на столбцы с высотой 1
        for i in range(len(bin_values)):
            if bin_values[i] == 0:
                bin_values[i] = (bin_values[i - 1] + bin_values[i + 1]) / 2

        # Рисуем заполненные столбцы
        plt.bar(bin_edges[:-1], bin_values, width=np.diff(bin_edges), align='edge', alpha=0.7, edgecolor='black', color='cyan')

        # Добавляем график аппроксимации
        sns.kdeplot(normal, color='red')

        plt.title(f'{"normal"}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)

        plt.axvline(x=LeftMathematicalExpectation - RightDispersion, color='purple', linewidth=2, clip_on=False, label='min \mu - max\sigma')
        plt.annotate(f'{LeftMathematicalExpectation - RightDispersion:.2f}', 
             xy=(LeftMathematicalExpectation - RightDispersion, 0), 
             xytext=(LeftMathematicalExpectation - RightDispersion, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(LeftMathematicalExpectation - RightDispersion, 0, marker='o', markersize=10, color='purple')

        plt.axvline(x=RightMathematicalExpectation + RightDispersion, color='purple', linewidth=2, clip_on=False, label='max \mu + max\sigma')
        plt.annotate(f'{RightMathematicalExpectation + RightDispersion:.2f}', 
             xy=(RightMathematicalExpectation + RightDispersion, 0), 
             xytext=(RightMathematicalExpectation + RightDispersion, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(RightMathematicalExpectation + RightDispersion, 0, marker='o', markersize=10, color='purple')

        plt.axvline(x=LeftMathematicalExpectation, color='red', linewidth=2, clip_on=False, label='min \mu')
        plt.annotate(f'{LeftMathematicalExpectation:.2f}', 
             xy=(LeftMathematicalExpectation, 0), 
             xytext=(LeftMathematicalExpectation, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(LeftMathematicalExpectation, 0, marker='o', markersize=10, color='red')


        plt.axvline(x=RightMathematicalExpectation, color='red', linewidth=2, clip_on=False, label='max \mu')
        plt.annotate(f'{RightMathematicalExpectation:.2f}', 
             xy=(RightMathematicalExpectation, 0), 
             xytext=(RightMathematicalExpectation, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(RightMathematicalExpectation, 0, marker='o', markersize=10, color='red')
        plt.legend()
        plt.savefig(f'{"normal"}, size {power}.png')

    drawInfoNormal(array_of_dispersion, array_of_mathematical_expectation, powers)


def pyasson_distribution(quantiles, powers):
    print("poisson_distribution:")
    bins = [10, 18]
    array_of_dispersion = []
    array_of_mathematical_expectation = []
    for power, bin in zip(powers, bins):

        poisson = np.random.poisson(10, power)

        average = np.mean(poisson)

        dispersion = np.std(poisson)

        variance = np.var(poisson)

        excess = (np.sum((poisson - average)**4) / (power * variance**2)) - 3 

        LeftMathematicalExpectation = average - dispersion * quantiles[0] / math.sqrt(power - 1)

        RightMathematicalExpectation = average + dispersion * quantiles[0] / math.sqrt(power - 1)

        LeftDispersion = dispersion * (1 - 0.5 * quantiles[0] * math.sqrt(excess + 2) / math.sqrt(power))

        RightDispersion = dispersion * (1 + 0.5 * quantiles[0] * math.sqrt(excess + 2) / math.sqrt(power))

        print(str(round(LeftMathematicalExpectation, 3)) + " < m < " + str(round(RightMathematicalExpectation, 3)))

        print(str(round(LeftDispersion, 3)) + " < s < " + str(round(RightDispersion, 3)))

        array_of_dispersion.append(LeftDispersion)
        array_of_dispersion.append(RightDispersion)
        array_of_mathematical_expectation.append(LeftMathematicalExpectation)
        array_of_mathematical_expectation.append(RightMathematicalExpectation)


        # Создаем гистограмму с заполненными столбцами
        plt.figure(figsize=(8, 6))
        bin_values, bin_edges, _ = plt.hist(poisson, bins=bin, density=True, alpha=0.7, edgecolor='black')

        # Замена пустых столбцов на столбцы с высотой 1
        for i in range(len(bin_values)):
            if bin_values[i] == 0:
                bin_values[i] = (bin_values[i - 1] + bin_values[i + 1]) / 2

        # Рисуем заполненные столбцы
        plt.bar(bin_edges[:-1], bin_values, width=np.diff(bin_edges), align='edge', alpha=0.7, edgecolor='black', color='cyan')

        # Добавляем график аппроксимации
        sns.kdeplot(poisson, color='red')

        plt.title(f'{"poisson"}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)

        plt.axvline(x=LeftMathematicalExpectation - RightDispersion, color='purple', linewidth=2, clip_on=False, label='min \mu - max\sigma')
        plt.annotate(f'{LeftMathematicalExpectation - RightDispersion:.2f}', 
             xy=(LeftMathematicalExpectation - RightDispersion, 0), 
             xytext=(LeftMathematicalExpectation - RightDispersion, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(LeftMathematicalExpectation - RightDispersion, 0, marker='o', markersize=10, color='purple')

        plt.axvline(x=RightMathematicalExpectation + RightDispersion, color='purple', linewidth=2, clip_on=False, label='max \mu + max\sigma')
        plt.annotate(f'{RightMathematicalExpectation + RightDispersion:.2f}', 
             xy=(RightMathematicalExpectation + RightDispersion, 0), 
             xytext=(RightMathematicalExpectation + RightDispersion, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(RightMathematicalExpectation + RightDispersion, 0, marker='o', markersize=10, color='purple')

        plt.axvline(x=LeftMathematicalExpectation, color='red', linewidth=2, clip_on=False, label='min \mu')
        plt.annotate(f'{LeftMathematicalExpectation:.2f}', 
             xy=(LeftMathematicalExpectation, 0), 
             xytext=(LeftMathematicalExpectation, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(LeftMathematicalExpectation, 0, marker='o', markersize=10, color='red')


        plt.axvline(x=RightMathematicalExpectation, color='red', linewidth=2, clip_on=False, label='max \mu')
        plt.annotate(f'{RightMathematicalExpectation:.2f}', 
             xy=(RightMathematicalExpectation, 0), 
             xytext=(RightMathematicalExpectation, -0.02), 
             ha='center', 
             fontsize=10)
        plt.plot(RightMathematicalExpectation, 0, marker='o', markersize=10, color='red')
        plt.legend()

        plt.savefig(f'{"poisson"}, size {power}.png')
    
    drawInfoPoisson(array_of_dispersion, array_of_mathematical_expectation, powers)


def main():
    powers = [20, 100]
    quantiles_normal20 = [2.086, 32.852, 8.907]
    quantiles_normal100 = [1.984, 128.422, 73.361]
    quantiles_poisson = [1.96]
    quantiles_normal = [quantiles_normal20, quantiles_normal100]
    calculate_normal(quantiles_normal, powers)
    pyasson_distribution(quantiles_poisson, powers)


if __name__ == "__main__":
    main()


