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
https://juejin.cn/post/7137219458635923492
'''
import bisect
import collections
import math
import queue
import sys
from typing import List

import cachetools

'''
7 整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围[−2^31, 2^31− 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-integer
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class Reverse:
    def reverse(self, x: int) -> int:
        INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
        result = 0
        num = 0
        while x != 0:
            if result < INT_MIN // 10 + 1 or result > INT_MAX // 10:
                return 0
            num = x % 10
            '''
            这里是为了处理负数问题
            '''
            if x < 0 and num > 0:
                num -= 10
                '''
                这里也是py的问题，除以10以后，会出现小数，因此要把尾巴切掉
                '''
            x = (x - num) // 10
            result = result * 10 + num

        return result


R = Reverse()
print(f'原始数据123,翻转后结果{R.reverse(123)}')
print(f'原始数据-456,翻转后结果{R.reverse(-456)}')
print(f'原始数据7890,翻转后结果{R.reverse(7890)}')

'''
37 解数独
编写一个程序，通过填充空格来解决数独问题。

数独的解法需 遵循如下规则：

数字1-9在每一行只能出现一次。
数字1-9在每一列只能出现一次。
数字1-9在每一个以粗实线分隔的3x3宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用'.'表示。

输入：board = [
["5","3",".",".","7",".",".",".","."],
["6",".",".","1","9","5",".",".","."],
[".","9","8",".",".",".",".","6","."],
["8",".",".",".","6",".",".",".","3"],
["4",".",".","8",".","3",".",".","1"],
["7",".",".",".","2",".",".",".","6"],
[".","6",".",".",".",".","2","8","."],
[".",".",".","4","1","9",".",".","5"],
[".",".",".",".","8",".",".","7","9"]]
输出：[
["5","3","4","6","7","8","9","1","2"],
["6","7","2","1","9","5","3","4","8"],
["1","9","8","3","4","2","5","6","7"],
["8","5","9","7","6","1","4","2","3"],
["4","2","6","8","5","3","7","9","1"],
["7","1","3","9","2","4","8","5","6"],
["9","6","1","5","3","7","2","8","4"],
["2","8","7","4","1","9","6","3","5"],
["3","4","5","2","8","6","1","7","9"]]
解释：输入的数独如上图所示，唯一有效的解决方案如下所示：

'''


class SolveSudoku:

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        init bitset
        '''
        cols = [0] * 9
        row = [0] * 9
        squar = [[0] * 3 for _ in range(3)]
        space = list()
        Valid = False
        for l1 in board:
            for l2 in l1:
                if l2 == '.':
                    space.append((l1, l2))
                else:
                    dig = int(board[l1, l2]) - 1
                    cols[l1] = cols[l1] ^ (1 << dig)
                    row[l2] = row[l2] ^ (1 << dig)
                    squar[l1 / 3, l2 / 3] ^= (1 << dig)


'''
42 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
示例 2：

输入：height = [4,2,0,3,2,5]
输出：9


提示：

n == height.length
1 <= n <= 2 * 10^4
0 <= height[i] <= 10^5

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/trapping-rain-water
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class Trap:
    def trap(self, height: List[int]) -> int:
        stack = list()
        sumVal = 0
        for idx, i in enumerate(height):
            while len(stack) > 0 and height[stack[-1]] < i:
                leftestValIdx = stack.pop()
                if not stack:
                    break
                cur = stack[-1]
                width = idx - cur - 1
                curHeight = min(height[cur], height[idx]) - height[leftestValIdx]
                sumVal += width * curHeight
            stack.append(idx)
        return sumVal


height = [4, 2, 0, 3, 2, 5]
s = Trap().trap(height)
print('雨水是:', s)

'''
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。



示例 1：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
示例 2：

输入：nums = [0,1]
输出：[[0,1],[1,0]]
示例 3：

输入：nums = [1]
输出：[[1]]


提示：

1 <= nums.length <= 6
-10 <= nums[i] <= 10
nums 中的所有整数 互不相同

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/permutations
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class Permute:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return self.process(nums)

    def process(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 1:
            return [nums]
        ret = []
        for i, val in enumerate(nums):
            rest = self.removeIdx(nums, i)
            p = self.process(rest)
            for p1 in p:
                p1.insert(0, val)
                ret.append(p1)
        return ret

    def removeIdx(self, nums: List[int], idx) -> List[int]:

        ret = []
        for i, val in enumerate(nums):
            if i != idx:
                ret.append(val)
        return ret


print('Permute结果', Permute().permute([1, 2, 3]))

'''
59.螺旋矩阵1
给你一个正整数n ，生成一个包含 1 到n2所有元素，且元素按顺时针顺序螺旋排列的n x n 正方形矩阵 matrix 。



示例 1：


输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
示例 2：

输入：n = 1
输出：[[1]]


提示：

1 <= n <= 20

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/spiral-matrix-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class GenerateMatrix:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        up, down, left, right = 0, n - 1, 0, n - 1
        tar = n * n
        num = 1
        while num <= tar:
            for i in range(left, right + 1):
                matrix[up][i] = num
                num += 1
            up += 1
            for i in range(up, down + 1):
                matrix[i][right] = num
                num += 1
            right -= 1
            for i in range(right, left - 1, -1):
                matrix[down][i] = num
                num += 1
            down -= 1
            for i in range(down, up - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
        return matrix

        return matrix


print('螺旋矩阵')
for m in GenerateMatrix().generateMatrix(3):
    print(m)

'''
给定一个数组 prices ，它的第i 个元素prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

示例 1：

输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
示例 2：

输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。


提示：

1 <= prices.length <= 105
0 <= prices[i] <= 104

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class MaxProfit:

    def maxProfit(self, prices: List[int]) -> int:
        minVal = sys.maxsize
        maxPro = 0
        for i in range(len(prices)):
            if prices[i] < minVal:
                minVal = prices[i]
            elif prices[i] - minVal > maxPro:
                maxPro = prices[i] - minVal
        return maxPro


print('MaxProfit', MaxProfit().maxProfit([2, 4, 1]))

'''
给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。


示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
示例 2：

输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1


提示：

1 <= envelopes.length <= 105
envelopes[i].length == 2
1 <= wi, hi <= 105

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/russian-doll-envelopes
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class MaxEnvelopes:

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        '''
        思路如下，先对envlopes的h做升序。
        如果h值一样，则对w做降序
        之后对所有的w做LIS，值就是套娃的值
        '''
        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        '''
        辅助数组end
        '''
        end = [envelopes[0][1]]
        for i in range(1, n):
            if (num := envelopes[i][1]) > end[-1]:
                end.append(num)
            else:
                idx = bisect.bisect_left(end, num)
                end[idx] = num
        return len(end)


class MaxEnvelopesTest:
    @staticmethod
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        n = len(envelopes)
        end = [envelopes[0][1]]
        for i in range(1, n):
            if envelopes[i][1] > end[-1]:
                end.append(envelopes[i][1])
            else:
                searchidx = bisect.bisect_left(end, envelopes[i][1])
                end[searchidx] = envelopes[i][1]
        return len(end)


print('MaxEnvelopes:', MaxEnvelopes().maxEnvelopes([[5, 4], [6, 4], [6, 7], [2, 3]]))

'''
给你一个大小为 m x n 的二进制矩阵 grid 。

岛屿是由一些相邻的1(代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设grid 的四个边缘都被 0（代表水）包围着。

岛屿的面积是岛上值为 1 的单元格的数目。

计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。



示例 1：


输入：grid = 
[
[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],
[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
示例 2：

输入：grid = [[0,0,0,0,0,0,0,0]]
输出：0


提示：

m == grid.length
n == grid[i].length
1 <= m, n <= 50
grid[i][j] 为 0 或 1

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/max-area-of-island
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


class MaxAreaOfIsland:
    cache = list(list())

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        h = len(grid)
        w = len(grid[0])
        maxIsland = 0
        for i in range(h):
            for j in range(w):
                '''
                表明这个岛屿之前被扫描过了
                '''
                maxIsland = max(self.search(i, j, grid), maxIsland)
        return maxIsland

    def search(self, x: int, y: int, grid: List[List[int]]) -> int:
        '''
        查找连通性
        '''
        if x < 0 or y < 0 or x == len(grid) or y == len(grid[0]):
            # 边界值
            return 0
        if grid[x][y] == 0:
            return 0

        grid[x][y] == 0
        island = self.search(x + 1, y, grid) + \
                 self.search(x - 1, y, grid) + \
                 self.search(x, y + 1, grid) + \
                 self.search(x, y - 1, grid) + 1

        return island


island = [
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
# print('最大联通岛屿：', MaxAreaOfIsland().maxAreaOfIsland(island))

'''
Alice 和 Bob 轮流从黑板上擦掉一个数字，Alice 先手。如果擦除一个数字后，剩余的所有数字按位异或运算得出的结果等于 0 的话，当前玩家游戏失败。另外，如果只剩一个数字，按位异或运算得到它本身；如果无数字剩余，按位异或运算结果为0。

并且，轮到某个玩家时，如果当前黑板上所有数字按位异或运算结果等于 0 ，这个玩家获胜。

假设两个玩家每步都使用最优解，当且仅当 Alice 获胜时返回 true。



示例 1：

输入: nums = [1,1,2]
输出: false
解释: 
Alice 有两个选择: 擦掉数字 1 或 2。
如果擦掉 1, 数组变成 [1, 2]。剩余数字按位异或得到 1 XOR 2 = 3。那么 Bob 可以擦掉任意数字，因为 Alice 会成为擦掉最后一个数字的人，她总是会输。
如果 Alice 擦掉 2，那么数组变成[1, 1]。剩余数字按位异或得到 1 XOR 1 = 0。Alice 仍然会输掉游戏。
示例 2:

输入: nums = [0,1]
输出: true

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/chalkboard-xor-game
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''

class XorGame(object):
    def xorGame(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums)%2 == 0:
            return True
        xor = 0
        for i in nums:
            xor ^= i
        return xor == 0
XorGame().xorGame([1,1,2])

'''
你的音乐播放器里有N首不同的歌，在旅途中，你的旅伴想要听 L首歌（不一定不同，即，允许歌曲重复）。请你为她按如下规则创建一个播放列表：

每首歌至少播放一次。
一首歌只有在其他 K 首歌播放完之后才能再次播放。
返回可以满足要求的播放列表的数量。由于答案可能非常大，请返回它模10^9 + 7的结果。



示例 1：

输入：N = 3, L = 3, K = 1
输出：6
解释：有 6 种可能的播放列表。[1, 2, 3]，[1, 3, 2]，[2, 1, 3]，[2, 3, 1]，[3, 1, 2]，[3, 2, 1].
示例 2：

输入：N = 2, L = 3, K = 0
输出：6
解释：有 6 种可能的播放列表。[1, 1, 2]，[1, 2, 1]，[2, 1, 1]，[2, 2, 1]，[2, 1, 2]，[1, 2, 2]
示例 3：

输入：N = 2, L = 3, K = 1
输出：2
解释：有 2 种可能的播放列表。[1, 2, 1]，[2, 1, 2]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/number-of-music-playlists
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''

class NumMusicPlaylists:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        pass



'''
给出一个单词数组words，其中每个单词都由小写英文字母组成。

如果我们可以不改变其他字符的顺序，在 wordA的任何地方添加 恰好一个 字母使其变成wordB，那么我们认为wordA是wordB的 前身 。

例如，"abc"是"abac"的 前身，而"cba"不是"bcad"的 前身
词链是单词[word_1, word_2, ..., word_k]组成的序列，k >= 1，其中word1是word2的前身，word2是word3的前身，依此类推。
一个单词通常是 k == 1 的 单词链。

从给定单词列表 words 中选择单词组成词链，返回 词链的最长可能长度 。


示例 1：

输入：words = ["a","b","ba","bca","bda","bdca"]
输出：4
解释：最长单词链之一为 ["a","ba","bda","bdca"]
示例 2:

输入：words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
输出：5
解释：所有的单词都可以放入单词链 ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].
示例3:

输入：words = ["abcd","dbqca"]
输出：1
解释：字链["abcd"]是最长的字链之一。
["abcd"，"dbqca"]不是一个有效的单词链，因为字母的顺序被改变了。


提示：

1 <= words.length <= 1000
1 <= words[i].length <= 16
words[i]仅由小写英文字母组成。


来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-string-chain
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''
class LongestStrChain:
    def longestStrChain(self, words: List[str]) -> int:
        words.sort(key=lambda x:len(x))
        maxLength=0
        for w in range(len(words)):
            length=0
            for s in range(w,len(words)):
                if self.isPre(words[w],words[s]):
                    length+=1
            maxLength = max(length,maxLength)
        return maxLength

    def isPre(self, first:str, second:str) -> bool:
        '''
        判断两个字符串，前者是不是后者的前身
        '''
        if len(first) == len(second):
            return False
        isBig=False
        lastPos=0
        n= len(second)
        for f in first:
            #如果每一个字符都在其后面，那说明就是前身
            try:
                curIdx = second.index(f,lastPos,n)
            except:
                return False
            isBig =  curIdx >= lastPos
            if isBig:
                lastPos = curIdx
            else:
                return False
        return isBig

# word=["a","b","ba","bca","bda","bdca"]
word=["xbc","pcxbcf","xb","cxbc","pcxbc"]
print('LongestStrChain',LongestStrChain().longestStrChain(word))


'''
输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。



示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释:连续子数组[4,-1,2,1] 的和最大，为6。


提示：

1 <=arr.length <= 10^5
-100 <= arr[i] <= 100

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''
class MaxSubArray:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) <=0:
            return 0
        dp=nums[0]
        maxVal = nums[0]
        for i in range(1,len(nums)):
            ## 如果dp[i-1] 小于=0，那么久应该是用nums[i]来计算
            ## 如果dp[i-1]大于0，则应该把nums[i]加上
            dp = max(nums[i],dp+nums[i])
            maxVal = max(maxVal,dp)
        return maxVal

print('MaxSubArray:',MaxSubArray().maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


'''
0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。



示例 1：

输入: n = 5, m = 3
输出:3
示例 2：

输入: n = 10, m = 17
输出:2


限制：

1 <= n<= 10^5
1 <= m <= 10^6

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''

class LastRemaining:
    def lastRemaining(self, n: int, m: int) -> int:
        cycle= collections.deque()
        for i in range(n):
            cycle.append(i)
        count:int = 0

        while len(cycle)>1:
            count = count + 1
            val = cycle.popleft()
            if count == m:
                count = 0
            else:
                cycle.append(val)
        return cycle[0]
print('LastRemaining',LastRemaining().lastRemaining(5,3))

sys.setrecursionlimit(100000)
class LastRemaining2:
    def lastRemaining(self, n: int, m: int) -> int:
        if n ==0 :
            return 0
        x = self.lastRemaining(n-1,m)
        return (m+x)%n
print('LastRemaining2',LastRemaining2().lastRemaining(5,3))



'''
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
示例 1:

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
限制：

0 <= 数组长度 <= 10^5
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''
class MaxProfit:
    def maxProfit(self, prices: List[int]) -> int:
        minVal = sys.maxsize
        maxP = 0
        for i in range(len(prices)):
            if prices[i] < minVal:
                minVal = prices[i]
            if (prices[i]-minVal)>maxP:
                maxP = prices[i]-minVal
        return maxP
print('MaxProfit',MaxProfit().maxProfit([7,1,5,3,6,4]))


'''
给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

 

示例 1：

输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
示例 2：

输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。
 

提示：

1 <= s.length <= 1000
s 仅由小写英文字母组成

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-palindromic-subsequence
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''

from functools import lru_cache
class LongestPalindromeSubseq:
    cache={}
    def longestPalindromeSubseq(self, s: str) -> int:
        return self.palindromeSubseq(s,0,len(s)-1)
    @lru_cache()
    def palindromeSubseq(self,s:str, left:int,right:int) -> int:
        maxVal = 0
        'base case'
        if left == right:
            return 1
        if left+1 == right:
            if s[left] == s[right]:
                return 2
            return 1

        '左边右移动1步'

        p1= self.palindromeSubseq(s,left+1,right)

        '右边左移动1步'
        p2 = self.palindromeSubseq(s,left,right-1)
        '如果两遍的数字一样，那么至少目前存在2个回文，因此需要再加上各自缩减一步以后的值'
        p3 = self.palindromeSubseq(s,left+1,right-1)+2 if s[left] == s[right] else 0
        '如果两遍的数字都不一样，都缩减异步'
        p4 = self.palindromeSubseq(s,left+1,right-1)
        maxVal = max(max(p1,p2),max(p3,p4))
        return maxVal

print('LongestPalindromeSubseq:',LongestPalindromeSubseq().longestPalindromeSubseq("aaaaaaaaaaaaaaaaaaaaa"))

class LongestPalindromeSubseqDP:
    cache={}
    def longestPalindromeSubseq(self, s: str) -> int:
        return self.palindromeSubseq(s)
    def palindromeSubseq(self,s:str) -> int:
        N = len(s)
        dp = [[0]*N for _ in range(N)]
        'base case 边界'
        '状态转移，i是l, j 是r'
        '根据base case 所有相等的位置，都是1'
        '循环到N-1的位置，目的是为了一遍循环搞出来，因为i 和i+1的位置也是base case' \
        '提前把dp[N-1][N-1]的位置设置了，这样才可以保证数组不越界，从而一把循环搞定'
        dp[N-1][N-1] = 1
        for i in range(len(dp)-1,-1,-1):
            dp[i][i]=1
            for j in range(i+1,len(dp)):
                '''p1：self.palindromeSubseq(s,left+1,right)
                '''
                p1 = dp[i+1][j]
                '''p2:self.palindromeSubseq(s,left,right-1)
                '''
                p2 = dp[i][j-1]
                '''p3:self.palindromeSubseq(s,left+1,right-1)'''
                '''在这里，可以用记忆化搜索，因为在算前一步的时候，P3的位置曾经参与了P1，P2的运算，P1,P2不可能碧这个P3小。因此无需计算P3'''
                p3 = dp[i+1][j-1]
                '''记忆化搜索，既然无需考虑P3，那么只要P4大，就可以了，因此无需再进行判断直接计算P4，不用做相等判断'''
                p4 = dp[i+1][j-1]+2 if s[i] == s[j] else 0
                dp[i][j] = max(max(p1,p2),max(p3,p4))
        'end case''因为答案要的是0到n-1的位置上的结果。因此直接返回0,N-1的结果'
        return dp[0][N-1]


print('LongestPalindromeSubseqDP:',LongestPalindromeSubseqDP().longestPalindromeSubseq('aaaaaaaaaaaaaaaaaaaaa'))
print('LongestPalindromeSubseqDP:1a233b2c1d=',LongestPalindromeSubseqDP().longestPalindromeSubseq('1a233b2c1d'))
print('LongestPalindromeSubseqDP:',LongestPalindromeSubseqDP().longestPalindromeSubseq(
    'euazbipzncptldueeuechubrcourfpftcebikrxhybkymimgvldiwqvkszfycvqyvtiwfckexmowcxztkfyzqovbtmzpxojfofbvwnncajvrvdbvjhcrameamcfmcoxryjukhpljwszknhiypvyskmsujkuggpztltpgoczafmfelahqwjbhxtjmebnymdyxoeodqmvkxittxjnlltmoobsgzdfhismogqfpfhvqnxeuosjqqalvwhsidgiavcatjjgeztrjuoixxxoznklcxolgpuktirmduxdywwlbikaqkqajzbsjvdgjcnbtfksqhquiwnwflkldgdrqrnwmshdpykicozfowmumzeuznolmgjlltypyufpzjpuvucmesnnrwppheizkapovoloneaxpfinaontwtdqsdvzmqlgkdxlbeguackbdkftzbnynmcejtwudocemcfnuzbttcoew'))
'''
1. 可以读通讯稿的组数
校运动会上，所有参赛同学身上都贴有他的参赛号码。某班参赛同学的号码记于数组nums中。假定反转后的号码称为原数字的「镜像号码」。
如果两位同学满足条件：镜像号码 A + 原号码 B = 镜像号码 B + 原号码 A，则这两位同学可以到广播站兑换一次读通讯稿的机会，
为同班同学加油助威。请返回所有参赛同学可以组成的可以读通讯稿的组数，并将结果对10^9+7取余。
注意：

镜像号码中如存在前置零，则忽略前置零。
同一位同学可有多次兑换机会。

示例 1：

输入：nums = [17,28,39,71]
输出：3
解释：
共有三对同学，分别为 [17,28]、[17,39]、[28,39]。其中：
第一对同学：17 + 82 = 71 + 28；
第二对同学：17 + 93 = 71 + 39；
第三对同学：28 + 93 = 82 + 39。

示例 2：

输入：nums = [71, 60]
输出：1
解释：
共有一对同学，为 [71, 60]。
因为 71 + 6 = 17 + 60，此处 60 的镜像号码为 6，前导零被忽略。

提示：

0 <= nums.length <= 10^6
0 <= nums[i] <= 10^9



作者：干掉自己
链接：https://juejin.cn/post/7137219458635923492
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

'''
2.池塘计数
最近的降雨，使田地中的一些地方出现了积水，field[i][j]表示田地第i行j列的位置有：

若为W, 表示该位置为积水；
若为., 表示该位置为旱地。

已知一些相邻的积水形成了若干个池塘，若以W为中心的八个方向相邻积水视为同一片池塘。
请返回田地中池塘的数量。
示例 1：

输入:field = [".....W",".W..W.","....W.",".W....","W.WWW.",".W...."]
输出：3
提示：

1 <= field.length, field[i].length <= 100
field 中仅包含 W 和 .
作者：干掉自己
链接：https://juejin.cn/post/7137219458635923492
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

'''

3.数字默契考验
某数学兴趣小组有 N 位同学，编号为 0 ~ N-1，老师提议举行一个数字默契小测试：首先每位同学想出一个数字，按同学编号存于数组numbers。每位同学可以选择将自己的数字进行放大操作，每次在以下操作中任选一种（放大操作不限次数，可以不操作）：

将自己的数字乘以 2
将自己的数字乘以 3

若最终所有同学可以通过操作得到相等数字，则返回所有同学的最少操作次数总数；否则请返回 -1。

示例 1：
输入：numbers = [50, 75, 100]
输出：5
解释：
numbers[0] * 2 * 3 = 300 操作两次；
numbers[1] * 2 * 2 = 300 操作两次；
numbers[2] * 3 = 300 操作一次。共计操作五次。
复制代码
示例 2：
输入：numbers = [10, 14]
输出：-1
解释：无法通过操作得到相同数字。
复制代码
提示：

1 <= numbers.length <= 10^5
1 <= numbers[i] <= 10^9

作者：干掉自己
链接：https://juejin.cn/post/7137219458635923492
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

'''
