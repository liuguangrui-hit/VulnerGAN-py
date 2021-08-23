import joblib


data = []
with open('../data/cic_2017/adver_sets/knn_adver_example.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        try:
            data.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
        except:
            continue

model_path = "saved_model/"
model_p = model_path + "knn.m"
clf = joblib.load(model_p)

result = clf.predict(data)
sum = 0
wrong = 0
for i in result:
    sum += 1
    # print(i)
    if i == 0:
        wrong += 1
print(f"knn_Bypass ratio:{wrong / sum}")


