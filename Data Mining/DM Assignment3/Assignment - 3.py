#!/usr/bin/env python
# coding: utf-8

# In[1]:


file1 = open("D:/Study/MSIT/2nd Year/DataScience_2019501007/Data Mining/DM Assignment3/more_stats202_logs.txt")
content = list(file1.readlines())
file1.close()

count_a = 0
count_b = 0
count_ab = 0
count = 0


for i in content:
    count += 1
    if ("65.57.245.11") in i:
        if ("Mozilla/5.0 (X11; U; Linux i686 (x86_64); en-US; rv:1.8.1.3) Gecko/20070309 Firefox/2.0.0.3" in i):
            count_ab += 1
        else:
            count_a += 1
    elif ("Mozilla/5.0 (X11; U; Linux i686 (x86_64); en-US; rv:1.8.1.3) Gecko/20070309 Firefox/2.0.0.3" in i):
        count_b += 1
    else:
        continue

print("Count of A: ", count_a)
print("Count of B: ", count_b)
print("Count of AB: ", count_ab)
print("Total Transactions: ", count)

support_a = count_a / count
support_b = count_b / count
support_ab = count_ab / count
confidence_ab = support_ab / support_a
print("--------------------------------------")
print("Support of A:", support_a)
print("Support of B:", support_b)
print("Support of AB:", support_ab)
print("Confidence of AB:", confidence_ab)


# In[ ]:




