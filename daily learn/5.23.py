# -*- coding:utf-8 -*-
# @Time      :2022/5/23 10:27
# @Author    :Riemanner
# Write code with comments !!!
###二叉树的路径遍历
####c层序遍历加null

# class Treenode():
#     def __init__(self,val):
#         self.val=val
#         self.left=None
#         self.right=None
# a=Treenode(1)
# b=Treenode(2)
# c=Treenode(3)
# d=Treenode(5)
# e=Treenode(6)
# a.left=c
# a.right=b
# c.left=d
# b.right=e
# def cengxubianli(root):
#     res=[]
#     queue=[root]
#     while queue.count(None)!=len(queue):
#         path=[]
#         ll=[]
#         for i in queue:
#             if i==None:
#                 path.append(0)
#             else:
#                 path.append(i.val)
#         res.append(path[:])
#         for node in queue:
#             if not node:
#                 ll.append(None)
#                 ll.append(None)
#             else:
#                 if node.left:
#                     ll.append(node.left)
#                 else:
#                     ll.append(None)
#                 if node.right:
#                     ll.append(node.right)
#                 else:
#                     ll.append(None)
#         queue=ll
#     for i in range(len(res)-1,-1,-1):
#         start_zhen=0
#         end_zhen=len(res[i])-1
#         while res[i][start_zhen]==0 or res[i][end_zhen]==0:
#             if res[i][start_zhen]==0:
#                 start_zhen+=1
#             if res[i][end_zhen]==0:
#                 end_zhen-=1
#         if end_zhen-start_zhen+1>=len(res[i-1]):
#             return end_zhen-start_zhen+1
#         else:
#             return 1
# print(cengxubianli(a))

###二分查找
# def erfenchazhao(nums,target):
#     start_zhen=0
#     end_zhen=len(nums)-1
#     while start_zhen<=end_zhen:
#         mid_zhen=(end_zhen+start_zhen)//2
#         if nums[mid_zhen]>target:
#             end_zhen=mid_zhen-1
#         elif nums[mid_zhen]<target:
#             start_zhen=mid_zhen+1
#         else:
#             return True
#     return False
# print(erfenchazhao([1,2,3,4,6,7],1))
# def findSubstringInWraproundString(p):
#     cnt = [0 for _ in range(26)]
#     i = 0  # 表示当前的位置
#     while i < len(p):
#         start, curLen = ord(p[i]) - ord('a'), 1###start是定位元素位置方便进行填数，定位首字母的位置进行判断，curlen则表示当前维护的长度
#         i += 1###对下一个字符进行判断
#         while i < len(p) and (ord(p[i]) - ord(p[i - 1])) % 26 == 1:###用ascc相差为一表示连续
#             curLen += 1####因为连续所以长度可以加一
#             i += 1
#         for j in range(min(26, curLen)):  # 更新cnt
#             if curLen - j > cnt[(start +j) % 26]:
#                 cnt[(start +j) % 26] = curLen - j
#     return sum(cnt)
# def majorityElement(nums):
#     nums.append(pow(10,9)+1)
#     nums.sort()
#     start_zhen = 0
#     end_zhen = 1
#     res=[]
#     while end_zhen < len(nums):
#         if nums[end_zhen] == nums[start_zhen]:
#             end_zhen += 1
#         elif nums[end_zhen] != nums[start_zhen]:
#             if end_zhen - start_zhen > (len(nums)-1) //3:
#                 res.append(nums[start_zhen])
#                 start_zhen = end_zhen
#                 end_zhen += 1
#             else:
#                 start_zhen = end_zhen
#                 end_zhen += 1
#     return res
# print(majorityElement(nums = [1]))
# import copy
#
# def frequencySort(s):
#     dic_c={}
#     for i in s:
#         if i in dic_c:
#             dic_c[i]+=1
#         else:
#             dic_c[i]=1
#     ss=[]
#     ss+=s
#     ss.sort(reverse=True,key=lambda x: (dic_c[x],ord(x)))
#     return ''.join(ss)
# print(frequencySort(""))
# def topKFrequent(nums,k):
#     dic_c={}
#     for i in nums:
#         if i in dic_c:
#             dic_c[i]+=1
#         else:
#             dic_c[i]=1
#     nums.sort(reverse=True,key=lambda x:(dic_c[x],x))
#     while len(list(set(nums)))>k:
#         nums.pop()
#     return list(set(nums))
#
# print(topKFrequent([4,1,-1,2,-1,2,3],2))
# def topKFrequent(nums,k):
#     if len(list(set(nums)))==k:
#         dic_c={}
#         for i in nums:
#             dic_c[i]=dic_c.get(i,0)+1
#         nums=list(set(nums))
#         nums.sort(key=lambda x:dic_c[x])
#         return nums
#     else:
#         l = {}
#         for i in nums:
#             l[i] = l.get(i,0)+1
#         l = dict(sorted(l.items(), key=lambda item: item[1], reverse=True))
#         ll = []
#         k+=1
#         for key, value in l.items():
#             ll.append(key)
#             k -= 1
#             if k == 0:
#                 break
#         cc=[ll[-1],ll[-2]]
#         ll.pop()
#         ll.pop()
#         if l[cc[0]]>l[cc[1]]:
#             ll.append(cc[0])
#         elif l[cc[0]]<l[cc[1]]:
#             ll.append(cc[1])
#         else:
#             cc.sort()
#             ll.append(cc[0])
#         return ll
# print(topKFrequent(["the","day","is","sunny","the","the","the","sunny","is","is"],4))
# import collections
# def maxSlidingWindow(nums,k):
#     res = []
#     queue = collections.deque()
#     for i, num in enumerate(nums):
#         if queue and queue[0] == i - k:
#             queue.popleft()
#         while queue and nums[queue[-1]] < num:
#             queue.pop()
#         queue.append(i)
#         if i >= k - 1: # 遍历到第k个元素(第一个完整的滑动窗口)开始返回结果到结果集中
#             res.append(nums[queue[0]])
#     return res
# print(maxSlidingWindow(nums = [1,3,-1,-3,5,3,6,7], k = 3))
# def countDigitOne(n: int) -> int:
#     if n<=0:
#         return 0
#     ## 用数学的方法， 逐个计算，个位，十位，百位 等数位上的 "1" 的个数
#     count = 0
#     k = 1
#     while k <= n:
#         res=(n//(10*k))*k + min(max(n%(10*k)-k+1, 0), k)
#         count+=res
#         print(res)
#         k *= 10
#     return count
# countDigitOne(20)
###一位 0-26
###两位 27-26*26+26
###三位 703-26*26*26+26
# def convertToTitle(columnNumber: int) -> str:
#     letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     if columnNumber <= 26: return letter[columnNumber-1]
#     a, b = divmod(columnNumber-1, 26)
#     return convertToTitle(a) + letter[b]
# def containsNearbyDuplicate(nums,k) -> bool:
#     if len(nums)==1:
#         return False
#     elif len(nums)<=k:
#         if len(set(nums))<=k:
#             return True
#         else:
#             return False
#     else:
#         start_zhen=0
#         end_zhen=start_zhen+k
#         while end_zhen<len(nums):
#             if len(set(nums[start_zhen:end_zhen+1]))<=k:
#                 return True
#             else:
#                 start_zhen+=1
#                 end_zhen+=1
#         return False
# print(containsNearbyDuplicate([1,2,3,1],3))
# def containsNearbyAlmostDuplicate(nums,k,t):
#     if k==0:
#         return True
#     window = set()
#     for i, num in enumerate(nums):
#         if len(window) == k + 1:  ######先维护窗口
#             window.remove(nums[i - 1 - k])
#         if t==0:
#             if num in window:
#                 return True
#         else:
#             for j in window:
#                 if abs(j - nums[i]) <= t:
#                     return True
#         window.add(num)
#     return False
# print(containsNearbyAlmostDuplicate([1,5,9,1,5,9],2,3))
def findClosest(words,word1,word2):
    start_zhen=0
    res=len(words)
    while start_zhen<len(words):
        s=[word1,word2]
        if words[start_zhen] in s:
            s.remove(words[start_zhen])
            end_zhen=start_zhen+1
            while end_zhen<len(words):
                if words[end_zhen]==s[0]:
                    res=min(res,end_zhen-start_zhen)
                    start_zhen+=1
                    break
                else:
                    end_zhen+=1
            else:
                break
        else:
            start_zhen+=1
    return res
print(findClosest(["k","c","ur","jm","jh","dl","sa","qw","tqr","b","kl","hns","g","y","au","ksw","zy","mi","u","hne","n","ub","irq","na","k","sg","np","fi","hyd","p","aoi","ixp","ve","ll","yh","dh","qc","yg","ic","ef","ho","ueq","w","pb","b","bnd","ahe","jbf","gui","jsu","wo","m","pzj","g","o","xoa","l","uwm","kdp","ra","v","p","mq","s","cpm","f","ma","vyd","p","kzw","oa","k","qm","ifg","dlw","y","y","ml","adg","mkw","vjr","yxw","x","s","rfv","pb","w","rq","gun","aaf","x","jj","lp","lb","nk","q","xa","r","ku","ecq","m","zd","zra","zee","x","klx","tzb","lwe","d","y","r","u","o","y","n","ah","pcv","g","y","uj","pu","pyz","ee","gc","n","t","r","lhu","f","uw","h","gfc","s","j","ixi","zvk","uyy","ga","b","wzn","u","pst","vq","u","pdn","zsn","vxk","msn","in","ev","ozq","w","p","u","p","f","hg","iab","gu","a","cih","m","qai","mzs","ol","wu","xhm","sch","hf","kfw","iq","opa","t","g","ym","il","z","a","xw","noo","jxk","th","j","ifi","kx","nas","m","xvy","g","jcn","sg","t","g","hz","z","oc","kvy","j","x","t","vel","tf","vw","fvq","l","u","uml","ksy","tbl","xan","o","s","v","zhe","oo","u","bc","je","xo","k","ame","me","tux","to","vzk","v","k","khz","hng","cg","thb","qt","vez","x","gbh","d","csc","sc","vl","cky","zb","g","wn","snn","de","syl","rl","cah","p","nj","vs","u","id","zx","v","lb","w","qxg","urw","yt","q","dyo","yxo","fi","p","iyi","cyk","ys","ff","os","uuc","p","egr","dra","hb","cpi","rll","j","o","dez","zq","z","ny","hc","jq","cpz","ih","n","qo","xv","gm","rg","vfi","rj","apy","c","x","cca","y","w","bf","d","sj","iyp","qb","mb","p","sbc","q","gp","wrv","v","nt","xw","e","x","uvy","wgm","i","w","uyg","z","py","ybd","gew","uzp","y","a","bwd","a","h","hpa","fid","q","d","t","n","ik","gm","lo","suo","wfe","vaj","l","vkp","yw","v","jr","psn","bu","o","p","zf","ej","d","yan","x","x","tkw","xxy","ehr","b","ds","z","ncq","l","qm","qb","uzc","do","k","f","kz","je","r","b","aq","hz","k","ipv","v","bai","c","fu","s","pg","ctn","i","fw","vu","hej","xl","qtv","nn","wal","fd","iay","kf","t","vv","fu","b","z","udg","ypg","rx","e","wus","fh","c","f","q","ijv","vl","hh","po","sf","sl","t","acm","hp","m","z","rrx","r","b","na","g","bt","nmx","edo","gau","s","j","k","y","ph","fl","xv","n","hua","i","kzo","lgz","fpq","mvh","yf","jvc","out","uv","w","bpk","k","xx","gbn","kj","yq","z","ul","mz","dxr","onc","nfu","mla","kyw","n","v","l","nly","qz","t","kbz","bj","ovy","xmr","k","ugo","ri","s","wt","l","muf","k","b","gs","w","dj","vb","ieu","b","c","kj","vr","q","dy","udj","v","vwx","ny","m","if","xbr","yar","q","erl","wl","o","xsb","b","zx","gqs","jz","ozd","h","ny","ogm","qor","bg","her","hqt","qe","o","g","ov","iqv","p","p","cgh","oxx","j","m","ii","mw","itg","uo","i","ua","r","j","dch","wwb","nf","euf","em","x","huo","m","ro","quu","zl","i","tf","a","fx","x","kif","vx","l","rtx","kwf","w","yr","rkx","uur","m","ooz","co","dz","s","zs","ac","r","ty","jn","x","fti","j","tk","g","bff","p","dy","e","wr","tj","h","ee","bx","kw","rvs","xpz","yb","f","f","yym","hf","owh","mdz","thg","lb","f","erz","hjh","cy","tv","w","k","dsb","pa","j","q","pip","vmf","zet","k","gzs","pee","y","zgu","b","xf","pte","l","pq","pj","lzu","jwy","wgw","v","xfm","jyk","piy","gvo","pur","hzc","g","nvz","ox","kkr","do","kop","r","pd","ixk","y","qio","hf","yq","tnk","ga","g","dkj","yj","w","j","bl","e","g","ki","s","pwj","j","ju","sji","kh","mvq","hsh","k","d","qtq","rb","k","gd","n","xei","q","w","wz","esa","blf","kqk","l","bp","z","t","s","p","thx","jl","y","la","du","vdd","x","a","xhx","rp","hi","pb","b","z","aa","pug","us","tkt","y","w","tre","ie","mss","u","tg","dfj","h","ulo","dkp","o","bd","bqh","qx","fl","xm","a","uxm","nt","p","wc","tk","fr","sd","f","xj","eds","gc","xz","qqq","nfn","x","lm","q","ofr","jm","l","coh","pl","nx","x","yg","t","aip","zg","jtc","u","er","i","j","ph","z","j","ynt","wq","imb","gpe","til","ns","pyy","hq","qm","k","lp","o","j","vup","qfd","ohj","z","gg","bw","m","fii","fa","y","p","yaz","i","ig","of","p","ws","orp","arf","ru","urq","g","u","zg","zmh","wgx","l","b","bc","th","pe","d","juo","qq","jeu","w","j","yl","l","q","vki","al","n","hpb","pjo","ft","x","aal","hx","n","y","wgl","u","avy","l","wlw","bc","iik","hxj","icx","lp","qf","f","jay","eqs","nlf","yol","umb","s","ir","em","z","o","bip","f","syg","ep","wfy","ct","wuc","ccs","wsp","wej","en","g","bg","msi","yo","ba","s","iqw","mcs","kua","z","mwv","aa","tf","cvt","aox","q","my","g","h","gha","oz","g","l","iu","sza","td","pf","mi","mz","me","pt","bje","r","q","l","xcp","wz","bhc","sa","hq","or","qi","rv","x","vgx","q","es","fj","j","p","m","q","nqx","ay","hb","vn","km","zw","pxz","j","l","zx","aa","t","a","rr","glo","iqn","gm","s","nbu","e","pf","tfs","i","ly","rkv","a","pz","hl","okl","qfn","wr","zu","qg","a","a","dl","euz","lqi","egm","bgs","zv","bo","s","dx","m","r","xf","ij","gu","h","dm","qor","lne","ln","kz","s","ry","ml","n","kq","sz","nyx","m","s","pa","w","sbz","kxz","muz","bbw","fa","b","mb","oe","wve","tga","qi","re","hkf","jlj","vx","gg","glm","o","kvl","vvk","yfn","lt","c","kz","p","bq"],"bx","rx"))