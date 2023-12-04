# # def threeSumClosest(nums,target):
# #     if len(nums) == 3:
# #         return sum(nums)
# #     else:
# #         s = abs(target - (nums[0] + nums[1] + nums[2]))
# #         s1=(nums[0] + nums[1] + nums[2])
# #         for i in range(len(nums)):
# #             for j in range(i+1, len(nums)):
# #                 for k in range(j+1, len(nums)):
# #                     if abs(target - (nums[i] + nums[j] + nums[k])) < s:
# #                         s = abs(target - (nums[i] + nums[j] + nums[k]))
# #                         s1=(nums[i] + nums[j] + nums[k])
# #                         if s == 0:
# #                             print(s1)
# #         print(s1)
# # threeSumClosest([-1,2,1,-4],1)
# # def threeSumClosest(nums,target) -> int:int
# #     if len(nums) == 3:
# #         return sum(nums)
# #     else:
# #         nums = sorted(nums)
# #         s = abs(target - (nums[0] + nums[1] + nums[2]))
# #         s1 = nums[0] + nums[1] + nums[2]
# #         for i in range(len(nums)):
# #             for j in range(i + 1, len(nums)):
# #                 for k in range(j + 1, len(nums)):
# #                     if abs(target - (nums[i] + nums[j] + nums[k])) < s:
# #                         s = abs(target - (nums[i] + nums[j] + nums[k]))
# #                         s1 = nums[i] + nums[j] + nums[k]
# #                         if s == 0:
# #                             print(target)
# #                     if abs(target - (nums[i] + nums[j] + nums[k])) >= s:
# #                         continue
# #         print(s1)
# #
# # threeSumClosest([-1,2,1,-4],1)
# # def threeSumClosest(nums,target) -> int:
# #     if len(nums) == 3:
# #         return sum(nums)
# #     else:
# #         nums = sorted(nums)
# #         s = abs(target - (nums[0] + nums[1] + nums[2]))
# #         s1 = nums[0] + nums[1] + nums[2]
# #         for i in range(len(nums) - 2):
# #             for j in range(i + 1, len(nums) - 1):
# #                 if j >= 2 and nums[i] > 0 and nums[j] > 0 and abs(target - (nums[i] + nums[j] + nums[j + 1])) >= s:
# #                     continue
# #                     for k in range(j + 1, len(nums)):
# #                         if abs(target - (nums[i] + nums[j] + nums[k])) < s:
# #                             s = abs(target - (nums[i] + nums[j] + nums[k]))
# #                             s1 = nums[i] + nums[j] + nums[k]
# #                             if s == 0:
# #                                 print(target)
# #                         if abs(target - (nums[i] + nums[j] + nums[k])) >= s:
# #                         continue
# #         print(s1)
# # threeSumClosest([1,2,4,8,16,32,64,128],82)
# def threeSumClosest(nums,target) -> int:
#     if len(nums) == 3:
#         return sum(nums)
#     else:
#         nums = sorted(nums)
#         s = abs(target - (nums[0] + nums[1] + nums[2]))
#         s1 = nums[0] + nums[1] + nums[2]
#         for i in range(len(nums) - 2):
#             for j in range(i + 1, len(nums) - 1):
#                 for k in range(j + 1, len(nums)):
#                     if abs(target - (nums[i] + nums[j] + nums[k])) < s:
#                         s = abs(target - (nums[i] + nums[j] + nums[k]))
#                         s1 = nums[i] + nums[j] + nums[k]
#                         if s == 0:
#                             print(s1)
#                     elif abs(target - (nums[i] + nums[j] + nums[k])) >= s:
#                         continue
#
#         print(s1)
# def threeSumClosest(nums,target):
#     if len(nums) == 3:
#         return sum(nums)
#     else:
#         nums = sorted(nums)
#         ss = nums[0] + nums[1] + nums[2]
#         for i in range(len(nums) - 2):
#             star = i + 1
#             end = len(nums) - 1
#             while star < end:
#                 sss = nums[i] + nums[star] + nums[end]
#                 if abs(sss - target) < abs(ss - target):
#                     ss = sss
#                 if nums[end] > target-nums[i]:
#                     end -= 1
#                 if nums[star] < target-nums[i]:
#                     star += 1
#         print(ss)
# threeSumClosest([0,2,1,-3],1)
# def threeSumClosest(nums,target):
#     if len(nums) == 3:
#         return sum(nums)
#     else:
#         nums = sorted(nums)
#         ss = nums[0] + nums[1] + nums[2]
#         for i in range(len(nums) - 2):
#             s = nums[i]
#             star = i + 1
#             end = len(nums) - 1
#             while star < end:
#                 sss = nums[i] + nums[star] + nums[end]
#                 if abs(sss - target) < abs(ss - target):
#                     ss = sss
#                     if abs(sss - target) == 0:
#                         print(ss)
#                 if abs(target - s - nums[end] - nums[star]) >= abs(target - s - nums[end - 1] - nums[star]):
#                     end -= 1
#                 if abs(target - s - nums[end] - nums[star]) >= abs(target - s - nums[end] - nums[star + 1]):
#                     star += 1
#                 else:
#                     end-=1
#                     star+=1
#     print(ss)
# threeSumClosest([-1,0,1,1,55],3)
# def threeSumClosest(nums, target):
#     """
#     :type nums: List[int]
#     :type target: int
#     :rtype: int
#     """
#     if len(nums) == 3:
#         return sum(nums)
#     nums.sort()
#     ans = nums[0] + nums[1] + nums[2]
#     for i in range(len(nums) - 2):
#         start, end = i + 1, len(nums) - 1
#         while start < end:
#             tmp = nums[i] + nums[start] + nums[end]
#             if abs(tmp - target) < abs(ans - target):
#                 ans = tmp
#             if tmp > target:
#                 end -= 1
#             else:
#                 start += 1
#     return ans
#
# print(threeSumClosest([-1,0,1,1,55],3))
def fourSum(nums,target):
    if len(nums) < 4:
        return [[]]
    elif len(nums) == 4 and sum(nums) == target:
        return [nums]
    else:
        nums.sort()
        S = []
        S1 = {1, 2, 3, 4}
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                star = j + 1
                end = len(nums) - 1
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    continue
                while star < end:
                    if nums[i] + nums[j] + nums[star] + nums[end] == target:
                        if [nums[i], nums[j],nums[star],nums[end]] not in S :
                            S.append([nums[i], nums[j], nums[star], nums[end]])
                            end-=1
                            star+=1
                        else:
                            end-=1
                            star+=1
                    elif nums[i] + nums[j] + nums[star] + nums[end] != target:
                        if nums[i] + nums[j] + nums[star] + nums[end] > target:
                            end-=1
                        if nums[i] + nums[j] + nums[star] + nums[end] < target:
                            star+=1

    print(S)


fourSum([2,2,2,2,2],8)