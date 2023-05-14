'''
回文
'''
import copy
import time
import timeit
import cProfile
import operator
from functools import wraps
import pickle
from functools import partial
from operator import __mul__
from functools import lru_cache

def timeof(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        span = end - begin
        print(f'{func.__name__} 参数 ({args} {kwargs}) 耗时 {span} 秒')
        return result

    return wrapper


def rollback_wen():
    a = input("Enter The sequence")
    ispalindrome = a == a[::-1]
    print(ispalindrome)


'''
奇偶
'''


def odd_even():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    odd, even = [el for el in a if el % 2 == 1], [el for el in a if el % 2 == 0]
    print(odd, even)


def fib():
    a, b = 0, 1
    while b < 100:
        print(b)
        a, b = b, a + b





def singleton(cls):
    '''
    构造器
    '''
    instance = {}

    @wraps(cls)
    def warpper(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
            return instance[cls]

    return warpper


count = 0


@singleton
class sigtest:
    def __init__(self):
        global count
        count = count + 1

    pass


def exchange_two_num():
    a = 1
    b = 2
    a = a ^ b
    b = a ^ b
    a = a ^ b
    print(a, b)

    c, d = 3, 4
    c, d = d, c
    print(c, d)


def what_is_yield():
    print("starting...")
    while True:
        res = yield 4
        print("res:", res)


def what_is_yield_test():
    g = what_is_yield()
    print(next(g))
    print("*" * 20)
    print(next(g))
    print(next(g))


def dedup():
    items = [1, 2, 3, 4, 3, 2, 5, 4, 3, 6]
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def deepcopy_test():

    my_deep_copy = lambda obj: pickle.loads(pickle.dumps(obj))
    return my_deep_copy


class copyTest1:
    field = ''
    filed2 = ''


class copyTest:
    filed = ''
    filed2 = ''
    field3 = copyTest1()


c1 = copyTest()
deepcopy_test()("123")
testcp = copy.deepcopy(c1)


class ProtoTypeCopy(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.clone = lambda self, is_deep=True: \
            copy.deepcopy(self) if is_deep else copy.copy(self)


class Person(metaclass=ProtoTypeCopy):
    pass


p1 = Person()
p2 = p1.clone()
p3 = p1.clone(is_deep=False)
print(id(p1), type(p1))
print(id(p2), type(p2))
print(id(p3), type(p3))

'''迭代器'''


def fib(num):
    a, b = 0, 1
    for _ in range(num):
        a, b = b, a + b
        yield a


for a in fib(10):
    print(a)

'''multi'''


def multiply():
    '''
    上面代码的运行结果很容易被误判为[0, 100, 200, 300]。首先需要注意的是multiply函数用生成式语法返回了一个列表，
    列表中保存了4个Lambda函数，这4个Lambda函数会返回传入的参数乘以i的结果。需要注意的是这里有闭包（closure）现象，
    multiply函数中的局部变量i的生命周期被延展了，由于i最终的值是3，所以通过m(100)调列表中的Lambda函数时会返回300，
    而且4个调用都是如此。

    '''
    return [lambda x: i * x for i in range(4)]


print([m(100) for m in multiply()])


def multi_paral():

    return [partial(__mul__, i) for i in range(4)]


print([m(100) for m in multi_paral()])




@lru_cache()
def change_money(total):
    '''
    在上面的代码中，我们用lru_cache装饰器装饰了递归函数change_money，如果不做这个优化，
    上面代码的渐近时间复杂度将会是$O(3^N)$，而如果参数total的值是99，这个运算量是非常巨大的。
    lru_cache装饰器会缓存函数的执行结果，这样就可以减少重复运算所造成的开销，这是空间换时间的策略，也是动态规划的编程思想。
    '''
    if total == 0:
        return 1
    if total < 0:
        return 0
    return change_money(total - 2) + change_money(total - 3) + \
           change_money(total - 5)


print(timeit.timeit(stmt='change_money(4)', setup="from __main__ import change_money", number=100))


# print(ways)

@lru_cache()
def walk_ways(steps):
    '''
    还有一个非常类似的题目：“一个小朋友走楼梯，一次可以走1个台阶、2个台阶或3个台阶，
    问走完10个台阶一共有多少种走法？”，这两个题目的思路是一样，如果用递归函数来写的话非常简单。
    '''
    if steps == 0:
        return 1
    if steps < 0:
        return 0
    return walk_ways(steps - 1) + walk_ways(steps - 2) + walk_ways(steps - 3)


steps = walk_ways(10)
print("this child may step", steps)


@timeof
def show_spiral_matrix(n):
    '''
    写一个函数，给定矩阵的阶数n，输出一个螺旋式数字矩阵。
例如：n = 2，返回：

1 2
4 3
例如：n = 3，返回：

1 2 3
8 9 4
7 6 5
    '''
    matrix = [[0] * n for _ in range(n)]
    row, col = 0, 0
    num, direction = 1, 0
    while num <= n ** 2:
        if matrix[row][col] == 0:
            matrix[row][col] = num
            num += 1
        if direction == 0:
            if col < n - 1 and matrix[row][col + 1] == 0:
                col += 1
            else:
                direction += 1
        elif direction == 1:
            if row < n - 1 and matrix[row + 1][col] == 0:
                row += 1
            else:
                direction += 1
        elif direction == 2:
            if col > 0 and matrix[row][col - 1] == 0:
                col -= 1
            else:
                direction += 1
        else:
            if row > 0 and matrix[row - 1][col] == 0:
                row -= 1
            else:
                direction += 1
        direction %= 4
    for x in matrix:
        for y in x:
            print(y, end='\t')
        print()


show_spiral_matrix(6)

'''
运行上面的代码首先输出1 1 1，这一点大家应该没有什么疑问。接下来， 
通过Child1.x = 2给类Child1重新绑定了属性x并赋值为2，所以Child1.x会输出2，而Parent和Child2并不受影响。
执行Parent.x = 3会重新给Parent类的x属性赋值为3，由于Child2的x属性继承自Parent，所以Child2.x的值也是3；
而之前我们为Child1重新绑定了x属性，那么它的x属性值不会受到Parent.x = 3的影响，还是之前的值2
'''


class Parent:
    x = 1


class Child1(Parent):
    pass


class Child2(Parent):
    pass


print(Parent.x, Child1.x, Child2.x)
Child1.x = 2
print(Parent.x, Child1.x, Child2.x)
Parent.x = 3
print(Parent.x, Child1.x, Child2.x)


def init_new(x):
    class InnerCls:
        name = 10
        age = 30

        def say_hello(self):
            print(f'self.name={self.name}, self.age={self.age},x 是闭包传进来的值:{x}')

        def __init__(self):
            print('init_new_init', x)

        def __new__(cls, *args, **kwargs):
            print('init_new_new', x)
            return super().__new__(cls, *args, **kwargs)

    cls = InnerCls()
    return cls


innerCls = init_new(9)

innerCls.say_hello()




def eval_suffix(script):
    '''逆波兰式'''
    operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }
    stack = []
    for item in script.split():
        if item.isdigit():
            stack.append(float(item))
        else:
            item1 = stack.pop()
            item2 = stack.pop()
            stack.append(operators[item](item1, item2))
    return stack.pop()


cProfile.run('print(eval_suffix("2 3 4 * + 5 -"))')


def default_param(a, v=[]):
    v.append(a)
    return v


lista = default_param(10)
print(f'lista 的id{id(lista)}的结果是:{lista}')

listb = default_param(20, [])
print(f'listb 的id{id(listb)}的结果是:{listb}')
'''注意，虽然第三次调用，没有传参，但是由于第一次调用的时候，参数v已经初始化了，是一个隐藏的变量。
因此第三次调用的时候，虽然没有再赋值，但是会复用之前的隐藏变量
'''
listc = default_param('a')
print(f'lista 的id{id(lista)}的结果是:{lista}')
print(f'listc 的id{id(listc)}的结果是:{listc}')


def read_big_data(path):
    with open(path, 'rb') as files:
        for data in iter(lambda: files.read(8192), b''):
            # print(len(data))
            print(data)


read_big_data('/Users/baodan/develop/git/tensorflowlearn/MathForProgrammers/PCA.py')


class A:
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

obj = A(1)
obj.__value = 2
'''直接在obj.__value这种方式，并不会真的修改A类中self的__value，因为其真正的值是_A__value，而不是__value'''
print(obj.value)
print(obj.__value)
'''这样改，才是真的改变了A中self的__value'''
obj._A__value=2
print(obj.value)

'''排序'''
prices = {
    'AAPL': 191.88,
    'GOOG': 1186.96,
    'IBM': 149.24,
    'ORCL': 48.44,
    'ACN': 166.89,
    'FB': 208.09,
    'SYMC': 21.29
}
print(prices)
sorted_prices = sorted(prices,key=lambda x: prices[x], reverse=True)
print(sorted_prices)

'''命名规则'''
from collections import namedtuple
Card = namedtuple('Card',('age','name','gender'))
p1 = Card(13,'张三','M')
p2 = Card(25,'lisi','F')
class MyCard(Card):
    def show(self):
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return f'{self.name}{faces[self.age]}'
print(p1)
print(p2)
print('命名元组也是元组,里面也是不可变的')

'''写一个函数，传入的参数是一个列表（列表中的元素可能也是一个列表），
返回该列表最大的嵌套深度。例如：列表[1, 2, 3]的嵌套深度为1，列表[[1], [2, [3]]]的嵌套深度为3
'''


def max_deep(items):

    max_val = 0
    if isinstance(items, list) or isinstance(items,tuple) or isinstance(items,map):
        for i in items:
            max_val = max(max_deep(i)+1,max_val)
    return max_val

d = max_deep([1,2,(3,4,{5,6})])
print(d)


'''
要求：有一个通过网络获取数据的函数（可能会因为网络原因出现异常），写一个装饰器让这个函数在出现指定异常时可以重试指定的次数，
并在每次重试之前随机延迟一段时间，最长延迟时间可以通过参数进行控制。
'''

from functools import wraps
from random import random
from time import sleep
def retryer(*,retry_times=3,max_wait_sec=5,errors=(Exception,)):

    def decorate(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            for i in range(retry_times):
                try:
                    return func(*args,**kwargs)
                except errors:
                    sleep(random()*max_wait_sec)
            return None
        return wrapper
    return decorate


'''写一个函数实现字符串反转，尽可能写出你知道的所有方法。'''

def reverse(content):
    return content[::-1]

print(reverse("qwe123"))


'''列表中有1000000个元素，取值范围是[1000, 10000)，设计一个函数找出列表中的重复元素'''

def duplicate_elements(els):
    s = set()


'''
题目一：给你多条有序链表，先让你删除每条链表的倒数第N个节点（这个节点保证存在），
然后把所有链表合并成一个有序链表。
'''


'''
题目二：给你两条链表1和2，链表1中的部分节点的值在链表2中也能找到，请在链表1中删除这些节点
'''

'''
有N个城市（城市1，城市2.....城市N）以及若干条道路，道路的形式为(i,j)表示城市i和城市j之间有道路，
现在你可以最多修建2条路来保证城市1可以连通到城市N，修路的费用为(i-j)的平方，请给出最小的修路费用
'''


'''
做题：逆波兰表达式求值 ，面试官共享屏幕看题目
'''

'''
一道在1-1000里猜数字（一道求掷骰子的期望一道求坐飞机换座位次数的期望一道是信息传播的一道是选一定数量基金募资1、
猜数字具体在1-1000里猜数字若猜的数字比该数大损失x元比该数小损失y元不同xy下至少要准备多少钱才能保证猜对
'''

'''
飞机那道具体题目是：飞机是喝醉乘客的变体就是第一个乘客不知道自己的位置会坐乱坐坐到别人的位
'''

'''
第一题是一个打印二维数组的方块题目。
'''

'''
LC 
7 整数反转
37 解数独
42 接雨水
46 全排列
59.螺旋矩阵1
121. 买卖股票的最佳时机
354. 俄罗斯套娃信封问题
695. 岛屿的最大面积
810. 黑板异或游戏
920.播放列表的数量
1048.最长字符串链
剑指 Offer 42. 连续子数组的最
剑指 Offer 62. 圆圈中最后剩下
剑指 Offer 63. 股票的最大利润
'''