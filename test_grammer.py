a = {}
for i in range(10000000):
    a[str(i)] = i

for i in range(10000000):
    if str(i) in a:
       if (a[str(i)]) % 10000 == 0:
           print("hello")