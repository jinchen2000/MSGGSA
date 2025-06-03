import pandas as pd
import numpy as np
from itertools import combinations
from MSGGSA1 import BGSA  #


# 读取CSV文件
def load_data(feature_file, label_file):
    features = pd.read_csv(feature_file).values
    labels = pd.read_csv(label_file).values.flatten()
    return features, labels


# 对立学习生成对立种群
def generate_opposite_population(population, num_features):
    opposite_population = []
    for individual in population:
        opposite_individual = np.ones(num_features) - individual
        opposite_individual = (opposite_individual > 0.5).astype(int)  # 转换为二进制表示
        opposite_population.append(opposite_individual)
    return np.array(opposite_population)


# 合并特征子集
def merge_features(best_features_original, best_features_opposite):
    merged_features = set()
    for subset in best_features_original:
        merged_features.update(subset)
    for subset in best_features_opposite:
        merged_features.update(subset)
    return list(merged_features)


# 递归特征选择
def recursive_feature_selection(features, labels, num_features, K, generations, recursion_depth=0):
    # 初始化种群
    population_size = 100  # 种群大小为100
    population = np.random.randint(2, size=(population_size, num_features))
    opposite_population = generate_opposite_population(population, num_features)

    # 应用GSA算法
    gsa_original = BGSA(population, features, labels, generations=generations)
    gsa_original.run()
    best_individuals_original = gsa_original.get_top_k(K)

    gsa_opposite = BGSA(opposite_population, features, labels, generations=generations)
    gsa_opposite.run()
    pareto_front_opposite = gsa_opposite.get_pareto_front()

    # 提取特征子集
    best_features_original = [list(np.where(individual == 1)[0]) for individual in best_individuals_original]
    best_features_opposite = [list(np.where(individual == 1)[0]) for individual in pareto_front_opposite]

    # 打印当前递归深度信息
    print(f"Recursion Depth: {recursion_depth}")
    print(f"Best Features Original: {best_features_original}")
    print(f"Best Features Opposite: {best_features_opposite}")

    if recursion_depth < 5:
        # 合并特征并生成新的特征空间
        merged_features = merge_features(best_features_original, best_features_opposite)
        new_num_features = len(merged_features)
        new_features = features[:, merged_features]

        # 递归调用
        return recursive_feature_selection(new_features, labels, new_num_features, K, generations, recursion_depth + 1)
    else:
        # 最后一轮递归，提取最优解
        best_individual_original = best_individuals_original[0]
        best_individual_opposite = pareto_front_opposite[0] if pareto_front_opposite else best_individuals_original[0]

        best_solution = np.maximum(best_individual_original, best_individual_opposite)
        best_features = list(np.where(best_solution == 1)[0])

        print(f"Final Best Features: {best_features}")
        return best_features


# 主函数
if __name__ == "__main__":
    feature_file = 'fea_file'
    label_file = 'cla_file'
    K = 5  # 精英个体数量
    generations = 100  # GSA算法迭代次数

    features, labels = load_data(feature_file, label_file)
    num_features = features.shape[1]

    best_features = recursive_feature_selection(features, labels, num_features, K, generations)
    print(f"Best Features Selected: {best_features}")