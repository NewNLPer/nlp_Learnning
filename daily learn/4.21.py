####
# def removeDuplicates(nums):
#     n = len(nums)
#     if n == 1 or n == 2:
#         return n
#     else:
#         i = 0
#         if nums[0]!=nums[1]:
#             j=1
#         else:
#             j=2
#         k = 0
#         while j <= len(nums) - 1:
#             if nums[j] == nums[i]:
#                 nums[j] = 3 * pow(10,4) + 1
#                 k += 1
#                 j += 1
#             else:
#                 i = j
#                 if i<n-2 and nums[i]!=nums[i+1]:
#                     j = i + 1
#                 else:
#                     j=i+2
#         nums.sort()
#         return nums
# print(removeDuplicates([1,1,1,2,2,3]))
# def merge(nums1, m, nums2, n):
#     """
#     Do not return anything, modify nums1 in-place instead.
#     """
#     if n == 0:
#         return nums1
#     else:
#         num1_star=0
#         num2_star=0
#         k=0
#         while num2_star<len(nums2):
#             if nums2[num2_star]>=nums1[num1_star] and nums2[num2_star]<nums1[num1_star+1]:
#                 nums1[num1_star+2:m+1+k]=nums1[num1_star+1:m+k]
#                 num1_star+=1
#                 nums1[num1_star]=nums2[num2_star]
#                 num2_star+=1
#                 k+=1
#             elif nums2[num2_star]>=nums1[num1_star] and sum(nums1[num1_star+1:])==0:
#                 nums1[num1_star+1:]=nums2[num2_star:]
#                 return nums1
#             elif nums2[num2_star]<nums1[num1_star]:
#                 nums1[num1_star + 1:m + 1 + k] = nums1[num1_star:m + k]
#                 nums1[num1_star] = nums2[num2_star]
#                 num1_star+=1
#                 num2_star += 1
#                 k += 1
#             else:
#                 num1_star+=1
#         return nums1
# print(merge([0,0,0,0,0],0,[1,2,3,4,5],5))
def qiuhe(nums):
    star=0
    end=len(nums)-1
    tmp=0
    while star<=end:
        if nums[star]<=0:
            star+=1
        elif nums[end]<=0:
            end-=1
        else:
            tmp=max(sum(nums[star:end+1]),tmp)
            star+=1
            end-=1
    return tmp
print(qiuhe([-2,-1,-1]))
