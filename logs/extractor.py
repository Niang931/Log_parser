import re

with open('access.log', 'r') as f:
    logs = f.read().split('\n')

for log in logs:
    if log == "":
        logs.remove(log)
for log in logs:
    print(log)





# tokens_list = []
# for log in logs:
#     tokens = log.split()
#     if tokens:
#         tokens_list.append(tokens)
#
# log1 = tokens_list[:5]
# log2 = tokens_list[5:7]
# log3 = tokens_list[7:9]
# log4 = tokens_list[9:10]
# log5 = tokens_list[10:13]
# log6 = tokens_list[13:]
#
# for log in log1:
#     print(log)
#
# for log in log2:
#     print(log)
