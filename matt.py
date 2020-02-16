import matlab.engine

eng = matlab.engine.start_matlab()
a = [1, 2, 3]
a = matlab.int8(a)
b = eng.mattest(a)
print(b)
