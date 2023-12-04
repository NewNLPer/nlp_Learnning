# ##进行括号有效生成问题的进行，利用递归、
def permute(nums):
    if len(nums) == 1:
        return [nums]
    else:
        S = []
        if len(nums) == 2:
            return ([[nums[0], nums[1]], [nums[1], nums[0]]])
        else:
            for i in permute(nums[0:len(nums)-1]):
                for j in range(len(i) + 1):
                    sss=i[0:j]+[nums[-1]]+i[j:]
                    if sss not in S:
                        S.append(sss)
            return S

print(permute([1,1]))

