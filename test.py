from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist = []


# Domain类
class Domain:
    def __init__(self, _name, _label, _length, _numbers, _entropy):
        self.name = _name
        self.label = _label  # dga or not dga
        self.length = _length  # domain name length
        self.numbers = _numbers  # numbers in the domain name
        self.entropy = _entropy  # entropy of letters

    def returnData(self):
        return [self.length, self.numbers, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0


# 初始化，统计
def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(name)
            int_count = 0  # numbers
            entropy_1 = 0
            character_count = {}
            character_frequency = {}
            for i in name:
                if i.isdigit():
                    int_count += 1
                character_count[i] = name.count(i)
                character_frequency[i] = 1.0 * character_count[i] / length
                entropy_1 -= (character_count[i]) * math.log(character_count[i], 2)  # 计算熵

            numbers = int_count
            entropy = entropy_1
            domainlist.append(Domain(name, label, length, numbers, entropy))


def main():
    print("Initialize Raw Objects")
    initData("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix")
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    print(featureMatrix)
    print("Begin Training")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)

    test_file = open("test.txt")
    result_file = open("result.txt", "w")
    for line in test_file:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        name_test = line
        length_test = len(name_test)
        int_count_test = 0  # numbers
        entropy_1_test = 0
        character_count_test = {}
        character_frequency_test = {}
        for i in name_test:
            if i.isdigit():
                int_count_test += 1
            character_count_test[i] = name_test.count(i)
            character_frequency_test[i] = 1.0 * character_count_test[i] / length_test
            entropy_1_test -= (character_count_test[i]) * math.log(character_count_test[i], 2)  # 计算熵
        numbers_test = int_count_test
        entropy_test = entropy_1_test

        if clf.predict([[length_test, numbers_test, entropy_test]]) == 0:
            label_test = "notdga"
        else:
            label_test = "dga"
        content = line + "," + label_test
        result_file.write(content)
        result_file.write("\n")
    test_file.close()
    result_file.close()


if __name__ == '__main__':
    main()
