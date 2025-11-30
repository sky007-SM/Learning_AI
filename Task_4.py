#program to illustrate various set operations
Set1={1,2,3,4,5}
Set2={3,4,8,9,10}
print("Set1: ",Set1)
print("Set2: ",Set2)
#Union of Set1 and Set2
Union=Set1.union(Set2)
print("Union of Set1 and Set2: ",Union)
#Intersection of Set1 and Set2
Intersection=Set1.intersection(Set2)
print("Intersection of Set1 and Set2: ",Intersection)
#Difference of Set1 and Set2
Difference=Set1.difference(Set2)
print("Difference of Set1 and Set2: ",Difference)
#Symmetric difference of Set1 and Set2
SymmetricDifference=Set1.symmetric_difference(Set2)
print("Symmetric difference of Set1 and Set2: ",SymmetricDifference)







