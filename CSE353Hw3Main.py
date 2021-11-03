import numpy as  np
import matplotlib.pyplot as plt

##Assume that X as a list of lists where x = [[1, x, y], [1, x, y], ...] unless otherwise stated
##simple dot product between the w vector and another vector x
def dotProduct(w, x):
    total = 0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total

##generates a new w from input values
def newW(w, x, y, index):
    returnedW = w
    point = x[index]
    for i in range(len(w)):
        returnedW[i] = w[i] + y[index] * point[i]
    return returnedW

##checks if there is a mistakes with a set of elements x according to the boundary w
def hasMistake(w, x, y):
    for i in range(len(x)):
        point = x[i]
        dot = dotProduct(w, point)
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            return True
    return False

##counts the number of mistakes
def numMistakes(w, x, y):
    total = 0
    for i in range(len(x)):
        point = x[i]
        dot = dotProduct(w, point)
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            total += 1
    return total

##Perceptron learning algorithm for Linearly separable elements
def PLA(w, x, y):
    while (hasMistake(w, x, y)):
        for i in range(len(x)):
            point = x[i]
            dot = dotProduct(w, point)
            if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
                w = newW(w, x, y, i)
    return w

##Pocket perceptron learning algorithm for Linearly separable elements
def PocketPLA(w, x, y):
    mistakesCurrentW = numMistakes(w, x, y)
    for j in range(475):
        for i in range(len(x)):
            point = x[i]
            dot = dotProduct(w, point)
            if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
                tempW = newW(w, x, y, i)
                if mistakesCurrentW > numMistakes(tempW, x, y):
                    w = tempW

    return w

##creates the the list of lists x from the input list of strings
def getListOfX(input, stride, pointSize):
    tempX = []
    for i in range(len(input)):
        tempX.extend(input[i].split(','))
        tempX = [x.replace('\n', '') for x in tempX]
    for i in range(len(tempX)):
        tempX[i] = float(tempX[i])
    return groupX(tempX, stride, pointSize)

##helper method for getListOfX which groups the sub lists together forming each individual [1, x, y, ...]
def groupX(list, stride, pointSize):
    returnedX = []
    for i in range(stride):
        tempXPoint = []
        for j in range(pointSize):
            tempXPoint.append(list[i + j*stride])
        returnedX.append(tempXPoint)
    return returnedX

##gets the list of y boolean truth values
def getListOfY(input):
    tempY = []
    for i in range(len(input)):
        tempY.extend(input[i].split(','))
        tempY = [x.replace('\n', '') for x in tempY]
    for i in range(len(tempY)):
        tempY[i] = float(tempY[i])
    return tempY

##gets the list of x values with some boolean binary num which is either -1 or 1
def extractListOfX(x, y, num):
    listOfXValues = []
    for i in range(len(x)):
        if y[i] == num:
            point = x[i]
            listOfXValues.append(point[1])
    return listOfXValues

##gets the list of y values with some boolean binary num which is either -1 or 1
def extractListOfY(x, y, num):
    listOfYValues = []
    for i in range(len(x)):
        if y[i] == num:
            point = x[i]
            listOfYValues.append(point[2])
    return listOfYValues

##gets wither the largest x value or smallest value
def getExtremeX(x, largest):
    largestX = 0
    smallestX = 0
    for i in range(len(x)):
        point = x[i]
        if point[1] > largestX:
            largestX = point[1]
        if point[1] < smallestX:
            smallestX = point[1]
    if largest:
        return largestX
    return smallestX
##returns the y value from the linear equation derived from w
def getYFromW(w, x):
    return -1 * ((w[0] + (w[1] * x))/w[2])
##returns the error rate with boundary w applied on x
def errorRate(w, x, y):
    total = 0
    totalIncorrect = 0
    for i in range(len(x)):
        point = x[i]
        dot = dotProduct(w, point)
        total += 1
        if (dot < 0 and y[i] > 0) or (dot > 0 and y[i] < 0):
            totalIncorrect += 1
    return totalIncorrect / total


##Linear Separable data
with open('Data/X_LinearSeparable.txt') as f:
    X_LinearSeparable = f.readlines()
    f.close()
with open('Data/Y_LinearSeparable.txt') as f:
    Y_LinearSeparable = f.readlines()
    f.close()
##generates the x, y, and w
X_LinearSeparable = getListOfX(X_LinearSeparable, 20, 3)
Y_LinearSeparable = getListOfY(Y_LinearSeparable)
w_LinSep = PLA([0,1,1], X_LinearSeparable, Y_LinearSeparable)
##generates lists for plotting
xLinSep1 = extractListOfX(X_LinearSeparable, Y_LinearSeparable, 1)
yLinSep1 = extractListOfY(X_LinearSeparable, Y_LinearSeparable, 1)
xLinSepn1 = extractListOfX(X_LinearSeparable, Y_LinearSeparable, -1)
yLinSepn1 = extractListOfY(X_LinearSeparable, Y_LinearSeparable, -1)
smallestX = getExtremeX(X_LinearSeparable, True)
largestX = getExtremeX(X_LinearSeparable, False)
##creates the figure
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Q1 Linear Separable')
plt.plot(xLinSep1, yLinSep1, 'bo')
plt.plot(xLinSepn1, yLinSepn1, 'rx')
plt.plot([smallestX, largestX], [getYFromW(w_LinSep, smallestX), getYFromW(w_LinSep, largestX)])
#plt.show()
print("The error rate of PLA on Linear Separable elements is " + str(errorRate(w_LinSep, X_LinearSeparable, Y_LinearSeparable)))


##NonLinear Separable data
with open('Data/X_NonLinearSeparable.txt') as f:
    X_NonLinearSeparable = f.readlines()
    f.close()
with open('Data/Y_NonLinearSeparable.txt') as f:
    Y_NonLinearSeparable = f.readlines()
    f.close()
##generates the x, y, and w
X_NonLinearSeparable = getListOfX(X_NonLinearSeparable, 20, 3)
Y_NonLinearSeparable = getListOfY(Y_NonLinearSeparable)
w_NonLinSep = PocketPLA([0,1,1], X_NonLinearSeparable, Y_NonLinearSeparable)
##generates lists for plotting
xNonLinSep1 = extractListOfX(X_NonLinearSeparable, Y_NonLinearSeparable, 1)
yNonLinSep1 = extractListOfY(X_NonLinearSeparable, Y_NonLinearSeparable, 1)
xNonLinSepn1 = extractListOfX(X_NonLinearSeparable, Y_NonLinearSeparable, -1)
yNonLinSepn1 = extractListOfY(X_NonLinearSeparable, Y_NonLinearSeparable, -1)
smallestX = getExtremeX(X_NonLinearSeparable, True)
largestX = getExtremeX(X_NonLinearSeparable, False)
##creates the figure
plt.subplot(1, 3, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Q2 NonLinear Separable')
plt.plot(xNonLinSep1, yNonLinSep1, 'bo')
plt.plot(xNonLinSepn1, yNonLinSepn1, 'rx')
plt.plot([smallestX, largestX], [getYFromW(w_NonLinSep, smallestX), getYFromW(w_NonLinSep, largestX)])
##plt.show()
print("The error rate of Pocket PLA on NonLinear Separable elements is " + str(errorRate(w_NonLinSep, X_NonLinearSeparable, Y_NonLinearSeparable)))



##Handcrafted feature
with open('Data/X_Digits_HandcraftedFeature_Train.txt') as f:
    X_HandCraftedTraining = f.readlines()
    f.close()
with open('Data/Y_Digits_HandcraftedFeature_Train.txt') as f:
    Y_HandCraftedTraining = f.readlines()
    f.close()
with open('Data/X_Digits_HandcraftedFeature_Test.txt') as f:
    X_HandCraftedTest = f.readlines()
    f.close()
with open('Data/Y_Digits_HandcraftedFeature_Test.txt') as f:
    Y_HandCraftedTest = f.readlines()
    f.close()
##generates the x, y, and w
X_HandCraftedTraining = getListOfX(X_HandCraftedTraining, 1561, 3)
Y_HandCraftedTraining = getListOfY(Y_HandCraftedTraining)
X_HandCraftedTest = getListOfX(X_HandCraftedTest, 424, 3)
Y_HandCraftedTest = getListOfY(Y_HandCraftedTest)
w_HandCrafted = PocketPLA([1, 5, 3], X_HandCraftedTraining, Y_HandCraftedTraining)
##generates lists for plotting
xHandCrafted1 = extractListOfX(X_HandCraftedTest, Y_HandCraftedTest, 1)
yHandCrafted1 = extractListOfY(X_HandCraftedTest, Y_HandCraftedTest, 1)
xHandCraftedn1 = extractListOfX(X_HandCraftedTest, Y_HandCraftedTest, -1)
yHandCraftedn1 = extractListOfY(X_HandCraftedTest, Y_HandCraftedTest, -1)
smallestX = getExtremeX(X_HandCraftedTest, True)
largestX = getExtremeX(X_HandCraftedTest, False)
##creates the figure
plt.subplot(1, 3, 3)
plt.xlabel('Symmetry')
plt.ylabel('Mean Intensity')
plt.title('Q3 Handcrafted Feature')
plt.plot(xHandCrafted1, yHandCrafted1, 'bo')
plt.plot(xHandCraftedn1, yHandCraftedn1, 'rx')
plt.plot([smallestX, largestX], [getYFromW(w_HandCrafted, smallestX), getYFromW(w_HandCrafted, largestX)])
print("The error rate of Pocket PLA on Handcrafted elements is " + str(errorRate(w_HandCrafted, X_HandCraftedTest, Y_HandCraftedTest)))
# plt.show()

##Raw features
with open('Data/X_Digits_RawFeature_Train.txt') as f:
    X_Digits_RawFeature_Train = f.readlines()
    f.close()
with open('Data/Y_Digits_RawFeature_Train.txt') as f:
    Y_Digits_RawFeature_Train = f.readlines()
    f.close()
with open('Data/X_Digits_RawFeature_Test.txt') as f:
    X_Digits_RawFeature_Test = f.readlines()
    f.close()
with open('Data/Y_Digits_RawFeature_Test.txt') as f:
    Y_Digits_RawFeature_Test = f.readlines()
    f.close()
##generates the x, y, and w
X_Digits_RawFeature_Train = getListOfX(X_Digits_RawFeature_Train, 1561, 257)
Y_Digits_RawFeature_Train = getListOfY(Y_Digits_RawFeature_Train)
X_Digits_RawFeature_Test = getListOfX(X_Digits_RawFeature_Test, 424, 257)
Y_Digits_RawFeature_Test = getListOfY(Y_Digits_RawFeature_Test)
w_Temp = []
for i in range(257):##generates a temp w of all 1s
    w_Temp.append(1)
w_RawFeature = PocketPLA(w_Temp, X_Digits_RawFeature_Train, Y_Digits_RawFeature_Train)
print("The error rate of Pocket PLA on Raw Feature elements is " + str(errorRate(w_RawFeature, X_Digits_RawFeature_Test, Y_Digits_RawFeature_Test)))
plt.show()
