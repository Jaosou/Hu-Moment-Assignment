import math
import matplotlib.pyplot as plt
import numpy as np
frequency = 4
graph = []
graph2 = []

sinx = math.sin(2*math.pi*frequency*1/1000)

for i in range(0,1001):
    sum_graph1 = 0
    sinx = math.sin(2*math.pi*frequency*i/1000)
    sin4x = math.sin(2*math.pi*4*frequency*i/1000)
    sin6x = math.sin(2*math.pi*6*frequency*i/1000)
    sum_graph1 = sum_graph1 + sinx + (1/3)*sin4x + (1/5)*sin6x
    graph.append(sum_graph1)
    
    sum_graph2 = 0
    sin6x = math.sin(2*math.pi*6*frequency*i/1000)
    sum_graph2 = sum_graph2 + sinx + (1/5)*sin4x + (1/9)*sin6x
    graph2.append(sum_graph2)
    
# print(graph[1],sinx)
print(sinx)


plt.figure(figsize=(16,4))

# แสดงภาพ
plt.subplot(1,3,1)
plt.plot(graph)
plt.title("Graph of Sum of Sine Waves")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.subplot(1,3,2)
plt.plot(graph2)
plt.title("Graph of Sum of Sine Waves (Graph 2)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.show()