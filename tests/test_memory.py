from memory_profiler import profile

@profile
def my_func1():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    c = [3] * (2 * 10 ** 8)
    del b
    return a   

@profile
def my_func2():
    for _ in range(10):
      a = [1] * (10 ** 6)
      b = [2] * (2 * 10 ** 7)
      c = [3] * (2 * 10 ** 8)
    del b, c
    return a    
  
if __name__=='__main__':
    a = my_func1()
    my_func2()
