class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
def lena(head):
    if head is None:
        return 0
    else:
        h=1
        while head.next is not None:
            h+=1
            head=head.next
        return h
a=Node(1)
b=Node(2)
c=Node(3)
d=Node(4)
e=Node(5)
a.next=b
b.next=c
c.next=d
d.next=e
def yici(head):
    cur=head
    while cur.next.next is not None:
        cur=cur.next
    p1=cur.next
    cur.next=None
    p1.next=head
    return p1
