import pyfpgrowth

#僵尸主机频繁更换IP，很难通过ip确定僵尸主机
#使用FP-growth算法挖掘出来浏览器ua字段和被攻击的目标URL之间的关联关系，
#确定潜在的僵尸主机


#测试数据中ip表示攻击源ip
#ua表示浏览器的user-agent字段
#target表示被攻击的目标url
transactions=[]

with open("../../data/KnowledgeGraph/sample7.txt") as f:
    for line in f:
        line=line.strip('\n')
        ip,ua,target=line.split(',')
        print("Add (%s %s %s)" % (ip,ua,target))
        transactions.append([ip,ua,target])



patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)
rules = pyfpgrowth.generate_association_rules(patterns, 0.9)

print(rules)
