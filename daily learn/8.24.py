# -*- coding:utf-8 -*-
# @Time      :2022/8/24 18:16
# @Author    :Riemanner
# Write code with comments !!!

def function(n: int) -> int:
    if n<=0:
        return 0
    count = 0
    k = 1
    while k <= n:
        count += (n//(10*k))*k + min(max(n%(10*k)-k+1, 0), k)
        k *= 10
    return count