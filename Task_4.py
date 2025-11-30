#program to illustrate various set operations
Set1=[1,2,3,4,5]
Set2=[3,4,8,9,10]
print("Set1: ",Set1)
print("Set2: ",Set2)
#Union of Set1 and Set2
Union=[]
for i in Set1:
    if i not in Union:
        Union.append(i)
for i in Set2:
    if i not in Union:
        Union.append(i)
print("Union of Set1 and Set2: ",Union)
#Intersection of Set1 and Set2
Intersection=[]
for i in Set1:
    if i in Set2:
        Intersection.append(i)
print("Intersection of Set1 and Set2: ",Intersection)
#Difference of Set1 and Set2
Difference=[]
for i in Set1:
    if i not in Set2:
        Difference.append(i)
print("Difference of Set1 and Set2: ",Difference)
#Symmetric difference of Set1 and Set2
SymmetricDifference=[]
for i in Set1:
    if i not in Set2:
        SymmetricDifference.append(i)
for i in Set2:
    if i not in Set1:
        SymmetricDifference.append(i)
print("Symmetric difference of Set1 and Set2: ",SymmetricDifference)


