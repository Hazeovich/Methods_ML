import numpy as np

N = int(input())
Y = np.array([float(i) for i in input().split()])
fucking_func = lambda x: 8*np.sin(x) + 7*x + 5
#Я два дня подбирал эту "хорошую" функцию
#Задание супер странное, надо было назвать не экстрополяция, а подбор функции
#Перебрал и многочлен лагранжа и формулу ньютона уже не говоря о многочлене n-ой степени
#А решение максимально тупое, ставлю 0 из 10 этому квесту
iterator = int((Y[-5] - 5)/7)
while np.all(round(fucking_func(iterator),2) != round(Y[-1],2)):
    iterator+=0.5
print(fucking_func(iterator+1))