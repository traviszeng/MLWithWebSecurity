#FP-growth算法的hello world
import pyfpgrowth

transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]

#minSupport代表最小支持度
#minConf代表最低置信度
#patterns = pyfpgrowth.find_frequent_patterns(transactions, minSupport)
#rules = pyfpgrowth.generate_association_rules(patterns, minConf)


patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

print(patterns)
print(rules)
