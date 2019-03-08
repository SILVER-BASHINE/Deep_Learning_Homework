###################################

#请根据需求自己补充头文件、函数体输入参数。


###################################
#2 Vectorization
###################################

def vectorize_sumproducts(v1,v2):
    s=sum(np.multiply(v1,v2))
    return s

def vectorize_Relu(s):
    s=np.array(s)
    s[s<0]=0
    return s

def vectorize_PrimeRelu(s):
    s=np.array(s)
    s[s<=0]=0
    s[s>0]=1
    return s

######################################
#3 Variable length
######################################

#Slice

def Slice_fixed_point(s,length,startpoint):
    s=np.array(s)
    s_p=[]
    for dim in s:
        s_p.append(np.array([row[startpoint:startpoint+length] for row in dim]))
    return np.array(s_p)


def slice_last_point(s,endpoint):
    s=np.array(s)
    s_p=[]
    for dim in s:
        s_p.append(np.array([row[-endpoint:] for row in dim]))
    return np.array(s_p)

def slice_random_point(s,length):
    s=np.array(s)
    row_length=[[len(row) for row in dim] for dim in s]
    min_row=np.min(np.min(row_length))
    r=np.random.randint(min_row,size=1)
    s_p=[]
    for dim in s:
        temp=[]
        for row in dim:
            z=row[int(r):int(r)+length]
            temp.append(np.array(z))
        s_p.append(np.array(temp))
    return np.array(s_p)

#Padding

def pad_pattern_end(test):
    out1=[]
    i=0
    out2=[]
    row_length=[[len(row) for row in dim]for dim in test]
    max_row=np.max(np.max(row_length))
    col_length=[len(dim) for dim in test]
    max_col=np.max(col_length)
    for dim in test:
        temp=[]
        for row in dim:
            row=np.pad(row,(0,max_row-len(row)),'symmetric')
            row=list(row)
            temp.append(row)
        #print(temp)
        out1.append(temp)

    #print(out1)
    for dim in out1:
        dim=np.pad(dim,((0,max_col-len(dim)),(0,0)),'symmetric')
        dim_p=dim
        out2.append(dim_p)   
        
    return np.array(out2)

def pad_constant_central(test,cval):
    out1=[]
    out2=[]
    row_length=[[len(row) for row in dim]for dim in test]
    max_row=np.max(np.max(row_length))
    col_length=[len(dim) for dim in test]
    max_col=np.max(col_length)
    for dim in test:
        temp=[]
        for row in dim:
            if (max_row-len(row))%2==0:
                row=np.pad(row,(int((max_row-len(row))/2),int((max_row-len(row))/2)),'constant',constant_values=cval)
            else:
                row=np.pad(row,(int((max_row-len(row))/2),int((max_row-len(row))/2)+1),'constant',constant_values=cval)
            row=list(row)
            temp.append(row)
        out1.append(temp)

    for dim in out1:
        if (max_col-len(dim))%2==0:
            dim=np.pad(dim,((int((max_col-len(dim))/2),int((max_col-len(dim))/2)),(0,0)),'constant',constant_values=cval)
        else:
            dim=np.pad(dim,((int((max_col-len(dim))/2),int((max_col-len(dim))/2)+1),(0,0)),'constant',constant_values=cval)
        dim_p=dim
        out2.append(dim_p)
        
    return np.array(out2)

#######################################
#PyTorch
#######################################

# numpy&torch

def numpy2tensor():
    s=np.array(s)
    return torch.from_numpy(s)


def tesor2numpy=torch.FloatTensor(s)
    return s.numpy()

#Tensor Sum-products

def Tensor_Sumproducts(a,b):
    a=torch.FloatTensor(a)
    b=torch.FloatTensor(b)
    return a.dot(b)

#Tensor ReLu and ReLu prime

def Tensor_Relu():
    a=torch.FloatTensor(a)
    a[a<0]=0
    return a
    
def Tensor_Relu_prime():
    a=torch.FloatTensor(a)
    a[a<=0]=0
    a[a>0]=1
    return a