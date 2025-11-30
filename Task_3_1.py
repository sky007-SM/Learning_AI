#Program to print n numbers of fibonacci series
limit=int(input("Enter the limit: "))
if limit==1:
    print(0,end=" ")
elif limit==2:
    print(0,1,end=" ")
else:
    print(0,1,end=" ")
    a=0
    b=1
    for i in range(2,limit):
        c=a+b
        print(c,end=" ")
        a=b
        b=c
    print()