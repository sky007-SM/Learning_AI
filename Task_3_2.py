#Python program to print all positive numbers in a range
limit=int(input("Enter the limit: "))
List=[]
for i in range(limit):
    num=int(input("Enter the number: "))
    List.append(num)
print("The list is: ",List)
for i in List:
    if i > 0:
        print(i)




