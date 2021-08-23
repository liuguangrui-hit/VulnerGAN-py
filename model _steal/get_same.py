model_name_1 = 'NIDS_DNN' # 原模型
model_name_2 = 'NIDS_DNN_GRU' # 替代模型
file_1 = "same_record/" + model_name_1 + ".csv"
file_2 = "same_record/" + model_name_2 + ".csv"
label_1 = []
label_2 = []

with open(file_1, "r") as f:
    lines = f.readlines()
    for line in lines:
        try:
            v = line.strip("\n")
            # print(v)
            label_1.append(int(v))
        except:
            continue

with open(file_2, "r") as f:
    lines = f.readlines()
    for line in lines:
        try:
            v = line.strip("\n")
            # print(v)
            label_2.append(int(v))
        except:
            continue

cnt = len(label_1)
correct = 0
for i in range(cnt):
    if label_1[i] == label_2[i]:
        correct += 1
# print(len(label_1),len(label_2))
print("same ratio: {} / {} = {}".format(correct, cnt, 100. * correct / cnt))