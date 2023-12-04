# def plusOne(digits):
#     if digits[-1] != 9:
#         digits[-1] = digits[-1] + 1
#         return digits
#     else:
#         digits[-1] = 0
#         if len(digits)==1:
#             digits=[1,0]
#             return digits
#         for i in range(len(digits) - 2, -1, -1):
#             if digits[i] == 9:
#                 digits[i] = 0
#                 if i == 0:
#                     digits.insert(0, 1)
#                 continue
#             else:
#                 digits[i] = digits[i] + 1
#                 break
#         return digits
# print(plusOne([9]))
# def mySqrt(x):
#     #####采用二分查找
#     if x == 0 or x == 1:
#         return x
#     else:
#         List=list(range(100000))
#         i=0
#         j=len(List)-1
#         while i<j and j-i!=1:
#             mid=(i+j)//2
#             if x/List[mid]>=List[mid]:
#                 i=mid
#             else:
#                 j=mid
#         return i
# print(mySqrt(900))
def merge(nums1, m,nums2, n) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    # if n==0:
    #     return nums1
    # else:
    #     nums1[m:]=nums2
    #     nums1.sort()
    #     return nums1
#     if n == 0:
#         return nums1
#     elif m == 0:
#         nums1[m:] = nums2
#         return nums1
#     else:
#         num1_star = 0
#         num2_star = 0
#         k = 0
#         while num2_star < len(nums2):
#             if nums2[num2_star] >= nums1[num1_star] and nums2[num2_star] < nums1[num1_star + 1]:
#                 nums1[num1_star + 2:m + 1 + k] = nums1[num1_star + 1:m + k]
#                 num1_star += 1
#                 nums1[num1_star] = nums2[num2_star]
#                 num2_star += 1
#                 k += 1
#             elif nums2[num2_star] >= nums1[num1_star] and sum(nums1[num1_star + 1:]) == 0:
#                 nums1[num1_star + 1:] = nums2[num2_star:]
#                 return nums1
#             elif nums2[num2_star] < nums1[num1_star]:
#                 nums1[num1_star + 1:m + 1 + k] = nums1[num1_star:m + k]
#                 nums1[num1_star] = nums2[num2_star]
#                 num1_star += 1
#                 num2_star += 1
#                 k += 1
#             else:
#                 num1_star += 1
#         return nums1
# print(merge([0,0,0,0,0],0,[1,2,3,4,5],5))
def merge(nums1,m,nums2,n):
    if n == 0:
        return nums1
    elif m==0:
        nums1[m:]=nums2
        return nums1
    else:
            #########从后面到前面，进行排序
        num1_q=m-1
        num2_q=n-1
        k=m+n-1
        while num2_q>=0 and num1_q>=0:
            if nums1[num1_q]>nums2[num2_q]:
                nums1[k]=nums1[num1_q]
                k-=1
                num1_q-=1
            else:
                nums1[k]=nums2[num2_q]
                k-=1
                num2_q-=1
    if num1_q<0:
        nums1[0:k+1]=nums2[0:k+1]
    return nums1

print(merge([2,0],1,[1],1))