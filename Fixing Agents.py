#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import numpy as np
import sys
import random

node_columns = ["Node Id","X","Y"]
nodes_data = pd.read_csv("https://www.cs.utah.edu/~lifeifei/research/tpq/SF.cnode", header=None, delimiter=r"\s+")
nodes_data.columns = node_columns
nodes_data.head()

edge_columns = ["Edge Id","X","Y","Weight"]
edges_data = pd.read_csv("https://www.cs.utah.edu/~lifeifei/research/tpq/SF.cedge", header=None, delimiter=r"\s+")
edges_data.columns = edge_columns
edges_data.head()


# In[2]:


print("Nodes:",len(nodes_data))
print("Edges:",len(edges_data))


# In[4]:


def compute_dist_matrix(source_nodes,nodes,graph):
    dist_mat = {}
    for i in nodes:
        temp = {}
        for j in source_nodes:
            temp[j] = nx.dijkstra_path_length(graph,i,j)
        dist_mat[i] = temp
    return dist_mat

def gnn(nodes,tag_nodes,graph,dist_mat):
    dist = np.inf
    nearest_node = -1
    for i in tag_nodes:
        temp = 0
        for j in nodes:
            temp += dist_mat[i][j]
        if (temp < dist):
            dist = temp
            nearest_node = i
    return nearest_node

def nn(node,tag_nodes,graph,dist_mat):
    dist = np.inf
    nn = -1
    for i in tag_nodes:
        temp = nx.dijkstra_path_length(graph,i,node)
        if (temp < dist):
            dist = temp
            nn = i
    return nn

def gtp_using_gnn_2(source,A,B,destinations,graph):
    start = time.time()
    
    temp_list_nodes = []
    temp_list_nodes2 = []
        
    temp_list_nodes2 = B
    temp_list_nodes = A+B
    source_nodes = source + destinations
    dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
      
    p1 = gnn(source,A,graph,dist_mat)
    x = destinations
    x.append(p1)
    
    dist_mat_for_p1 = compute_dist_matrix([p1],temp_list_nodes2,graph)
    for i in temp_list_nodes2:
        dist_mat[i][p1] = dist_mat_for_p1[i][p1]
    p2 = gnn(x,B,graph,dist_mat)  
    
    b = len(source)
    sum1 = 0
    for i in range(b):
        sum1 += dist_mat[p1][source[i]]
    sum2 = 0
    sum2 += nx.dijkstra_path_length(graph,p1,p2)
    sum3 = 0
    for i in range(b):
        sum3 += dist_mat[p2][destinations[i]]
    path_cost = sum1 + b*sum2 + sum3
    path = [p1,p2]
    end = time.time()
    return path_cost,path,end-start


def gtp_using_gnn_3(source,A,B,C,destinations,graph):
    start = time.time()
    
    temp_list_nodes = []
    temp_list_nodes2 = []
    
    temp_list_nodes2 = C
    temp_list_nodes = A+B+C
    
    source_nodes = source + destinations
    dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
    
    p1 = gnn(source,A,graph,dist_mat)
    p2 = nn(p1,B,graph,dist_mat)
    x = destinations
    x.append(p2)
    
    dist_mat_for_p2 = compute_dist_matrix([p2],temp_list_nodes2,graph)
    for i in temp_list_nodes2:
        dist_mat[i][p2] = dist_mat_for_p2[i][p2]
    p3 = gnn(x,C,graph,dist_mat)
    
    b = len(source)
    sum1 = 0
    for i in range(b):
        sum1 += dist_mat[p1][source[i]]
    sum2 = 0
    sum2 += nx.dijkstra_path_length(graph,p1,p2)
    sum2 += nx.dijkstra_path_length(graph,p2,p3)
    sum3 = 0
    for i in range(b):
        sum3 += dist_mat[p3][destinations[i]]
    path_cost = sum1 + b*sum2 + sum3
    path = [p1,p2,p3]
    end = time.time()
    return path_cost,path,end-start

def gtp_using_gnn_4(source,A,B,C,D,destinations,graph):
    start = time.time()
    
    temp_list_nodes = []
    temp_list_nodes3 = []
    temp_list_nodes3 = D

    temp_list_nodes = A+B+C+D
    source_nodes = source + destinations    
    dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
    
    p1 = gnn(source,A,graph,dist_mat)
    p2 = nn(p1,B,graph,dist_mat)
    p3 = nn(p2,C,graph,dist_mat)
    x = destinations
    x.append(p3)
    dist_mat_for_p3 = compute_dist_matrix([p3],temp_list_nodes3,graph)
    
    for i in temp_list_nodes3:
        dist_mat[i][p3] = dist_mat_for_p3[i][p3]
    p4 = gnn(x,D,graph,dist_mat)
    
    b = len(source)
    sum1 = 0
    for i in range(b):
        sum1 += dist_mat[p1][source[i]]
    sum2 = 0
    sum2 += nx.dijkstra_path_length(graph,p1,p2)
    sum2 += nx.dijkstra_path_length(graph,p2,p3)
    sum2 += nx.dijkstra_path_length(graph,p3,p4)
    sum3 = 0
    for i in range(b):
        sum3 += dist_mat[p4][destinations[i]]
    path_cost = sum1 + b*sum2 + sum3
    path = [p1,p2,p3,p4]
    end = time.time()
    return path_cost,path,end-start

def gtp_using_gnn_5(source,A,B,C,D,E,destinations,graph):
    start = time.time()
    temp_list_nodes = []
    temp_list_nodes4 = []
    temp_list_nodes4 = E

    temp_list_nodes = A+B+C+D+E
    source_nodes = source + destinations    
    dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
    
    p1 = gnn(source,A,graph,dist_mat)
    p2 = nn(p1,B,graph,dist_mat)
    p3 = nn(p2,C,graph,dist_mat)
    p4 = nn(p3,D,graph,dist_mat)
    x = destinations
    x.append(p4)
    dist_mat_for_p4 = compute_dist_matrix([p4],temp_list_nodes4,graph)
    
    for i in temp_list_nodes4:
        dist_mat[i][p4] = dist_mat_for_p4[i][p4]
    p5 = gnn(x,E,graph,dist_mat)
    
    b = len(source)
    sum1 = 0
    for i in range(b):
        sum1 += dist_mat[p1][source[i]]
    sum2 = 0
    sum2 += nx.dijkstra_path_length(graph,p1,p2)
    sum2 += nx.dijkstra_path_length(graph,p2,p3)
    sum2 += nx.dijkstra_path_length(graph,p3,p4)
    sum2 += nx.dijkstra_path_length(graph,p4,p5)
    sum3 = 0
    for i in range(b):
        sum3 += dist_mat[p5][destinations[i]]
    path_cost = sum1 + b*sum2 + sum3
    path = [p1,p2,p3,p4,p5]
    end = time.time()
    return path_cost,path,end-start

def gtp_using_gnn_6(source,A,B,C,D,E,F,destinations,graph):
    start = time.time()
    temp_list_nodes = []
    temp_list_nodes5 = []
    temp_list_nodes5 = F

    temp_list_nodes = A+B+C+D+E+F
    source_nodes = source + destinations    
    dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
    
    p1 = gnn(source,A,graph,dist_mat)
    p2 = nn(p1,B,graph,dist_mat)
    p3 = nn(p2,C,graph,dist_mat)
    p4 = nn(p3,D,graph,dist_mat)
    p5 = nn(p4,E,graph,dist_mat)
    x = destinations
    x.append(p5)
    dist_mat_for_p5 = compute_dist_matrix([p5],temp_list_nodes5,graph)
    
    for i in temp_list_nodes5:
        dist_mat[i][p5] = dist_mat_for_p5[i][p5]
    p6 = gnn(x,F,graph,dist_mat)
    
    b = len(source)
    sum1 = 0
    for i in range(b):
        sum1 += dist_mat[p1][source[i]]
    sum2 = 0
    sum2 += nx.dijkstra_path_length(graph,p1,p2)
    sum2 += nx.dijkstra_path_length(graph,p2,p3)
    sum2 += nx.dijkstra_path_length(graph,p3,p4)
    sum2 += nx.dijkstra_path_length(graph,p4,p5)
    sum2 += nx.dijkstra_path_length(graph,p5,p6)
    sum3 = 0
    for i in range(b):
        sum3 += dist_mat[p6][destinations[i]]
    path_cost = sum1 + b*sum2 + sum3
    path = [p1,p2,p3,p4,p5,p6]
    end = time.time()
    return path_cost,path,end-start

def compute_dist_matrix_BF(nodes,graph):
    dist_mat = {}
    for i in nodes:
        temp = dict(nx.single_source_dijkstra_path_length(graph,i))
        dist_mat[i] = temp
    return dist_mat

def create_clusters(G,cluster_names,cluster_nodes):
    
    #print(cluster_names)
    
    cluster.clear()
    clusters_cnt.clear()
    
    for i in cluster_names:
        cluster.append(i)
    for k in cluster_nodes:
        clusters_cnt.append(k)
        
    j = 0
    for cl_cnt in clusters_cnt:
        i = 0
        while(i<cl_cnt):
            x = np.random.randint(len(nodes_data))
            if(G.nodes[x]["Role"]==""):
                G.nodes[x]["Role"]=cluster[j]
                i = i + 1
            else:
                continue
        j = j + 1
    
    return G

def create_s_d(sr_dt,G):
    #print("create_s_d() called ...")
    i = 0
    while(i<sr_dt):
        x = np.random.randint(len(nodes_data))
        if(G.nodes[x]["Role"]==""):
            G.nodes[x]["Role"]="Source"
            i = i + 1
        else:
            continue
            
    i = 0
    while(i<sr_dt):
        x = np.random.randint(len(nodes_data))
        if(G.nodes[x]["Role"]==""):
            G.nodes[x]["Role"]="Destination"
            i = i + 1
        else:
            continue
            
    return G

def traversal(sr_dt,clusters,cluster_cnt):
    
    clusters.append("Source")
    clusters.extend(cluster)
    clusters.append("Destination")
       
    cluster_cnt.append(sr_dt)
    cluster_cnt.extend(clusters_cnt)
    cluster_cnt.append(sr_dt)
    
    print("Inside traversal()",clusters)
    print("Inside traversal()",cluster_cnt)
    
    return clusters,cluster_cnt

def findMax(mat, N):
        maxElement = -sys.maxsize - 1
        for i in range(N):
            for j in range(N):
                if (mat[i][j] > maxElement):
                    maxElement = mat[i][j]
        return maxElement

def shortest_path(G, node):
    length, path = nx.single_source_dijkstra(G, node)
    return length, path

def graph_construction(G,clusters,cluster_cnt):
    import numpy as np
    import pandas as pd
    import sys
    import time
    
    clusters1 = []
    for i in clusters:
        if(i!="Source" and i!="Destination"):
            clusters1.append(i)
    
    n_w_r_c = 0
    for i in list(G.nodes):
        if(G.nodes[i]["Role"]!=""):
            n_w_r_c = n_w_r_c + 1
    print("Nodes Participating:",n_w_r_c)
    
    dict1 = {}
    path_weights = []
    dummy = {}
    
    st_time = time.time()
    
    for i in list(G.nodes):
        if(G.nodes[i]["Role"] in clusters1):
            #print("For Node: ",i," and Role: ",G.nodes[i]["Role"])
            dict1, dummy = shortest_path(G, i)
            path_weights.append(dict1)
    
    n_rols = []             # ADDING DESTINATION NODE OT CREATE K+2 PARTITE GRAPH
    for i in list(G.nodes):
        if(G.nodes[i]["Role"] in clusters1):
            n_rols.append(i)
    
    data_paths = pd.DataFrame(path_weights,index=n_rols)
    data_paths.to_csv("Graph11.csv")
    
    G1 = nx.Graph()
    
    for i in list(G.nodes):
        if(G.nodes[i]["Role"]!=""):
            G1.add_node(i)
            G1.nodes[i]["Role"] = G.nodes[i]["Role"]

    for i in list(G1.nodes):
        for j in n_rols:
            if(i!=j):
                for k in range(0,len(clusters)):
                    if(k+1<len(clusters)):
                        if(G1.nodes[i]["Role"]==clusters[k] and G1.nodes[j]["Role"]==clusters[k+1]):
                            G1.add_edge(i,j,weight=data_paths[i][j])

    for i in n_rols:
        for j in list(G1.nodes):
            for k in (len(clusters)-2,len(clusters)):
                if(k+1<len(clusters)):
                    if(G1.nodes[i]["Role"]==clusters[k] and G1.nodes[j]["Role"]==clusters[k+1]):
                        G1.add_edge(i,j,weight=data_paths[j][i])
    
    
    dict1 = {}
    for i in list(G1.nodes):
        ind_data = []
        for j in list(G1.nodes):
            if(G1.get_edge_data(i,j)!=None):
                ind_data.append(G1.get_edge_data(i,j)["weight"])
            else:
                ind_data.append(-1)
        dict1[i] = ind_data
    n_rols1 = []
    for i in G1.nodes():
        n_rols1.append(i)
   
    data = pd.DataFrame(dict1,index=n_rols1)
    data.to_csv("Graph21.csv")
    
    dict_1 = {}
    dict_11 = {}
    dict_paths_1 = []
    path_weights_1 = []
    a_1 = []
    for i in list(G1.nodes):
        if(G1.nodes[i]["Role"]==clusters[1]):
            a_1.append(i)
            dict_1, dict_11 = shortest_path(G1, i)
            path_weights_1.append(dict_1)
            dict_paths_1.append(dict_11)
        
    datapw = pd.DataFrame(path_weights_1,index=a_1)
    datapl = pd.DataFrame(dict_paths_1,index=a_1)
    
    #print("(K+2)-partite graph",nx.info(G1))

    clusters2 = [clusters[0],clusters[1],clusters[len(clusters)-2],clusters[len(clusters)-1]]
    G2 = nx.Graph()
    for i in list(G1.nodes):
        if(G1.nodes[i]["Role"] in clusters2):
            G2.add_node(i)
            G2.nodes[i]["Role"] = G1.nodes[i]["Role"]

    #print("4-partite graph",nx.info(G2))    
    f_two = []
    cnt = 0
    while(cnt<2):
        f_two.append(clusters[cnt])
        cnt = cnt + 1

    l_two = []
    cnt = len(clusters)-1
    while(cnt>=len(clusters)-2):
        l_two.append(clusters[cnt])
        cnt = cnt - 1

    for i in list(G2.nodes):
        for j in list(G2.nodes):
            if(i!=j):
                if(G2.nodes[i]["Role"]==f_two[0] and G2.nodes[j]["Role"]==f_two[1]):
                    G2.add_edge(i,j,weight=data[j][i])

    for i in list(G2.nodes):
        for j in list(G2.nodes):
            if(i!=j):
                if(G2.nodes[i]["Role"]==l_two[1] and G2.nodes[j]["Role"]==l_two[0]):
                    G2.add_edge(i,j,weight=data[j][i])

    
    source_weights = []
    
    for k in range(len(a_1)):
        source_weights=path_weights_1[k]
        #print(source_weights)
        for i in source_weights:
            if(G.nodes[a_1[k]]["Role"]==clusters2[1] and G.nodes[i]["Role"]==clusters2[2]):
                #print(a_1[k],i,source_weights[i])
                G2.add_edge(a_1[k],i,weight=source_weights[i])
    
    #print("final 4-partite graph",nx.info(G2))
    
    n_rols2 = []
    for i in list(G2.nodes):
        if(G2.nodes[i]["Role"] in clusters2):
            n_rols2.append(i)
    
    
    # G2.get_edge_data(i,j)["weight"]
    
    
    dict1 = {}
    for i in list(G2.nodes):
        ind_data = []
        for j in list(G2.nodes):
            if(G2.get_edge_data(i,j)!=None):
                ind_data.append(G2.get_edge_data(i,j)["weight"])
            else:
                ind_data.append(-1)
        dict1[i] = ind_data

    n_rols2 = []
    for i in list(G2.nodes):
        n_rols2.append(i)
    data_2 = pd.DataFrame(dict1,index=n_rols2)
    #print(data_2)
    data_2.to_csv("Graph31.csv")
    
    S = []
    D = []
    A = []
    DP = []
    for i in G2.nodes:
        if(G2.nodes[i]["Role"] == clusters[0]):
            S.append(i)
    for i in G2.nodes:
        if(G2.nodes[i]["Role"] == clusters[len(clusters)-1]):
            D.append(i)
    for i in G2.nodes:
        if(G2.nodes[i]["Role"] == clusters[1]):
            A.append(i)
    for i in G2.nodes:
        if(G2.nodes[i]["Role"] == clusters[len(clusters)-2]):
            DP.append(i)
    dict2 = {}
    
    for i in A:
        for j in DP:
            result = []
            for k in range(len(S)):
                result.append(data_2[S[k]][i] + data_2[i][j] + data_2[j][D[k]]) 
            dict2[str(i)+":"+str(j)] = result
            
    data_3 = pd.DataFrame(dict2,index=S)
    data_3.head()
    data_3.to_csv("Graph41.csv")
    
    s_index = list(data_3.columns)
    
    data_3.to_csv("final.csv")
    col_sums = {}
    opt_sum = []
    for i in s_index:    
        opt_sum.append(data_3[i].sum())
        col_sums[i] = data_3[i].sum()
    
    OPT_path_length = min(opt_sum) 
    print("optimal path length A:", OPT_path_length)
    
    en_time = time.time()
    
    temp = OPT_path_length
    res = [key for key in col_sums if col_sums[key] == temp]
  
    # printing result 
    print("Keys with minimum values are : " + str(res))
    
    og = int(res[0].split(":")[0])
    dt = int(res[0].split(":")[1])
    
    print(datapl[dt][og])
    
    s_index = list(data_3.columns)
    size = len(s_index)
    flag = 0
    p = 0
    pareto=[]
    while( p != size):
        q =0
        while(q != size):
            if(q == p):
                q=q+1
                continue
            flag1 = 1
            for s in S:
                if(data_3[s_index[p]][s] >= data_3[s_index[q]][s]):
                    flag = 0
                else:
                    flag = 1
                    break
                if(data_3[s_index[p]][s] > data_3[s_index[q]][s]):
                    flag1 = 0
            if(flag == 0 and flag1 == 0):
                size = size-1
                data_3 = data_3.drop([s_index[p]],axis=1)
                s_index.remove(s_index[p])
                p = p - 1
                break
            else:
                if( q == ( size -1 ) ):
                    pareto.append(s_index[p])
                if(q == ( (size) - 2 ) and p == ( (size) -1 )  ):
                    pareto.append(s_index[p])
            q = q+1
        p = p+1
     

    M  = np.zeros( (len(S), len(S)), np.int64)
    maximum =[]
    for p in pareto:
        for b1,elem1 in enumerate(S):
            for b2,elem2 in enumerate(S):
                if(b2 > b1):
                    M[b1][b2] = abs(data_3[p][elem1] - data_3[p][elem2])
        maximum.append(findMax(M,len(S)))

    val = (pd.Series(maximum).idxmin())    
    eps = pareto[val]

    PO_path_with_epsilon_envy = 0
    for i in S:
        PO_path_with_epsilon_envy += data_3[eps][i]
    #print(PO_path_with_epsilon_envy)
    
    data_3.to_csv("final.csv")
    col_sums = {}
    opt_sum = []
    for i in s_index:    
        opt_sum.append(data_3[i].sum())
        col_sums[i] = data_3[i].sum()

    OPT_path_length = min(opt_sum)
    #print(OPT_path_length)
    price_of_fairness = (PO_path_with_epsilon_envy/OPT_path_length)
    print("PRICE OF FAIRNESS: ",price_of_fairness)
    
    return G,price_of_fairness,OPT_path_length, np.abs(en_time-st_time)

def brute_force_2(source,A,B,destinations,graph):
    start = time.time()
    b = len(source)
    dict_tag_to_node = {}
    temp_list_nodes = []   
    #print("before",A,B)
    for each in A:
        temp_list_nodes.append(each)
    for each in B:
        temp_list_nodes.append(each)
        
    #print("testing:", temp_list_nodes)

    dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)
    min_path_cost = np.inf  
    min_path = []
    
    
    for m in A:     
        for n in B:    
            sum1 = 0
            for i in range(0,b):
                sum1 += dist_mat[m][source[i]]
            sum2 = 0
            sum2 += dist_mat[m][n]
            sum3 = 0
            for i in range(0,b):
                sum3 += dist_mat[n][destinations[i]]
            total_cost = sum1 + b*sum2 + sum3
            if (total_cost < min_path_cost):
                min_path_cost = total_cost
                min_path = [m,n]
    end = time.time() 
    return min_path_cost,min_path,end-start

def brute_force_3(source,A,B,C,destinations,graph):
    start = time.time()
    b = len(source)
    temp_list_nodes = []
    #print("before",A,B,C)
    for each in A:
        temp_list_nodes.append(each)
    for each in B:
        temp_list_nodes.append(each)
    for each in C:
        temp_list_nodes.append(each)
        
    #print("testing:", temp_list_nodes)
    dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)
    min_path_cost = np.inf
    min_path = []   
    for m in A:       
        for n in B: 
            for p in C: 
                sum1 = 0
                for i in range(0,b):
                    sum1 += dist_mat[m][source[i]]
                sum2 = 0
                sum2 += dist_mat[m][n]
                sum2 += dist_mat[n][p]
                sum3 = 0
                for i in range(0,b):
                    sum3 += dist_mat[p][destinations[i]]
                total_cost = sum1 + b*sum2 + sum3
                if (total_cost < min_path_cost):
                    min_path_cost = total_cost
                    min_path = [m,n,p]
    end = time.time()
    return min_path_cost,min_path,end-start

def brute_force_4(source,A,B,C,D,destinations,graph):
    start = time.time()
    b = len(source)
    
    temp_list_nodes = []
     
    for each in A:
        temp_list_nodes.append(each)
    for each in B:
        temp_list_nodes.append(each)
    for each in C:
        temp_list_nodes.append(each)
    for each in D:
        temp_list_nodes.append(each)
    
            
    dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)

    min_path_cost = np.inf
    min_path = []
    start = time.time()
    for m in A:
        for n in B:
            for p in C:
                for q in D:
                    sum1 = 0
                    for i in range(0,b):
                        sum1 += dist_mat[m][source[i]]
                    sum2 = 0
                    sum2 += dist_mat[m][n]
                    sum2 += dist_mat[n][p]
                    sum2 += dist_mat[p][q]
                    sum3 = 0
                    for i in range(0,b):
                        sum3 += dist_mat[q][destinations[i]]
                    total_cost = sum1 + b*sum2 + sum3
                    if (total_cost < min_path_cost):
                        min_path_cost = total_cost
                        min_path = [m,n,p,q]
    end = time.time()
    return min_path_cost,min_path,end-start

def brute_force_5(source,A,B,C,D,E,destinations,graph):
    start = time.time()
    
    b = len(source)
    
    temp_list_nodes = []
    
    temp_list_nodes = A+B+C+D+E
             
    dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)

    min_path_cost = np.inf
    min_path = []
    for m in A:
        for n in B:
            for p in C:
                for q in D:
                    for r in E:
                        sum1 = 0
                        for i in range(0,b):
                            sum1 += dist_mat[m][source[i]]
                        sum2 = 0
                        sum2 += dist_mat[m][n]
                        sum2 += dist_mat[n][p]
                        sum2 += dist_mat[p][q]
                        sum2 += dist_mat[q][r]
                        sum3 = 0
                        for i in range(0,b):
                            sum3 += dist_mat[r][destinations[i]]
                        total_cost = sum1 + b*sum2 + sum3
                        if (total_cost < min_path_cost):
                            min_path_cost = total_cost
                            min_path = [m,n,p,q,r]
    end = time.time()
    return min_path_cost,min_path,end-start

def brute_force_6(source,A,B,C,D,E,F,destinations,graph):
    start = time.time()
    b = len(source)
    
    temp_list_nodes = []
    
    temp_list_nodes = A+B+C+D+E+F
             
    dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)

    min_path_cost = np.inf
    min_path = []
    start = time.time()
    for m in A:
        for n in B:
            for p in C:
                for q in D:
                    for r in E:
                        for t in F:
                            sum1 = 0
                            for i in range(0,b):
                                sum1 += dist_mat[m][source[i]]
                            sum2 = 0
                            sum2 += dist_mat[m][n]
                            sum2 += dist_mat[n][p]
                            sum2 += dist_mat[p][q]
                            sum2 += dist_mat[q][r]
                            sum2 += dist_mat[r][t]
                            sum3 = 0
                            for i in range(0,b):
                                sum3 += dist_mat[t][destinations[i]]
                            total_cost = sum1 + b*sum2 + sum3
                            if (total_cost < min_path_cost):
                                min_path_cost = total_cost
                                min_path = [m,n,p,q,r,t]
    end = time.time()
    return min_path_cost,min_path,end-start

import time

import networkx as nx
import pandas as pd
import numpy as np
import sys

G = nx.Graph()

for i in list(nodes_data["Node Id"]):
    #print(i," ",nodes_data["X"][i]," ",nodes_data["Y"][i])
    G.add_node(i, pos=(nodes_data["X"][i], nodes_data["Y"][i]))
    G.nodes[i]["Id"] = i
    G.nodes[i]["Role"] = ""
    
for i in list(edges_data["Edge Id"]):
    G.add_edge(edges_data["X"][i], edges_data["Y"][i], weight=edges_data["Weight"][i])


print(nx.info(G))

cluster = []
clusters_cnt = []

clusters = []
cluster_cnt = []

sr_dt = 5

graph_with_s_d = create_s_d(sr_dt,G)

cluster_names = ['A','B','C','D','E','F']
cluster_nodes = [5,10,15,20,25,30]
cluster_count = [2,3,4,5,6]


price_of_fairness = []
op_path = []
op_path_br = []
times2 = []
clsts = []
POF2 = []

opt = {}
ops = {}
ops_br = {}
ops_gnn = {}
times1 = {}
times1_br = {}
times1_gnn = {}
cnt = 0

seeds = [100*i for i in range(100)]
start = 0 
end = 1

for cnt in range(start,end):
    np.random.seed(seeds[cnt])
    random.seed(seeds[cnt])
    print(cnt+1,end=' ')
    price_of_fairness = []
    op_path = []
    op_path_br = []
    op_path_gnn = []
    time1 = []
    time1_br = []
    time1_gnn = []
    for ack in cluster_count:
        cluster_nm = cluster_names[0:ack]
        cluster_nd = cluster_nodes[0:ack]
        GWDS = create_clusters(graph_with_s_d,cluster_nm,cluster_nd)
        clusters,cluster_cnt = traversal(sr_dt,clusters,cluster_cnt)
        Gr, pof, op, top = graph_construction(GWDS,clusters,cluster_cnt)
        op_path.append(op)
        price_of_fairness.append(pof)
        time1.append(top)
        
        Source = []
        A = []
        B = []
        C = []
        D = []
        E = []
        F = []
        Destination = []

        for i in list(Gr.nodes()):
                if(Gr.nodes[i]["Role"] =='A'):
                    A.append(i) 

        for j in list(Gr.nodes()):
                if(Gr.nodes[j]["Role"] =='B'):
                    B.append(j)

        for k in list(Gr.nodes()):
                if(Gr.nodes[k]["Role"] =='C'):
                    C.append(k) 

        for l in list(Gr.nodes()):
                if(Gr.nodes[l]["Role"] =='D'):
                    D.append(l)
        
        for l1 in list(Gr.nodes()):
                if(Gr.nodes[l1]["Role"] =='E'):
                    E.append(l1)
        
        for l2 in list(Gr.nodes()):
                if(Gr.nodes[l2]["Role"] =='F'):
                    F.append(l2)
                    

        for m in list(Gr.nodes()):
                if(GWDS.nodes[m]["Role"] =="Source"):
                    Source.append(m)

        for n in list(Gr.nodes()):
                if(Gr.nodes[n]["Role"] =="Destination"):
                    Destination.append(n)

        if(ack==2):
            min_path_cost_br,min_path_br,runtime_br = brute_force_2(Source,A,B,Destination,Gr)
            min_path_cost_gnn,min_path_gnn,runtime_gnn = gtp_using_gnn_2(Source,A,B,Destination,Gr)
        elif(ack==3):
            min_path_cost_br,min_path_br,runtime_br = brute_force_3(Source,A,B,C,Destination,Gr)
            min_path_cost_gnn,min_path_gnn,runtime_gnn = gtp_using_gnn_3(Source,A,B,C,Destination,Gr)
        elif(ack==4):
            min_path_cost_br,min_path_br,runtime_br = brute_force_4(Source,A,B,C,D,Destination,Gr)
            min_path_cost_gnn,min_path_gnn,runtime_gnn = gtp_using_gnn_4(Source,A,B,C,D,Destination,Gr)
        elif(ack==5):
            min_path_cost_br,min_path_br,runtime_br = brute_force_5(Source,A,B,C,D,E,Destination,Gr)
            min_path_cost_gnn,min_path_gnn,runtime_gnn = gtp_using_gnn_5(Source,A,B,C,D,E,Destination,Gr)
        elif(ack==6):
            min_path_cost_br,min_path_br,runtime_br = brute_force_6(Source,A,B,C,D,E,F,Destination,Gr)
            min_path_cost_gnn,min_path_gnn,runtime_gnn = gtp_using_gnn_6(Source,A,B,C,D,E,F,Destination,Gr)
        
        print("Bruteforce:",min_path_cost_br)
        op_path_br.append(min_path_cost_br)
        time1_br.append(runtime_br)

        print("GNN:",min_path_cost_gnn)
        op_path_gnn.append(min_path_cost_gnn)
        time1_gnn.append(runtime_gnn)
        
        for node in list(G.nodes):
            if(G.nodes[node]["Role"] in cluster_names):
                G.nodes[node]["Role"]=""
        
        clusters.clear()
        cluster_cnt.clear()
        
    opt[cnt] = price_of_fairness
    ops[cnt] = op_path
    times1[cnt]=time1

    ops_br[cnt] = op_path_br
    times1_br[cnt] = time1_br
    
    ops_gnn[cnt] = op_path_gnn
    times1_gnn[cnt] = time1_gnn    
    
    PF2 = pd.DataFrame(opt,index=cluster_count)
    PF2.to_csv("POF_source_destination_fixed_"+str(start)+"_"+str(end)+"_run"+".csv")

    rPF2 = pd.DataFrame(times1,index=cluster_count)
    rPF2.to_csv("runningtime_source_destination_fixed_"+str(start)+"_"+str(end)+"_run"+".csv")
    
    rPF2_br = pd.DataFrame(times1_br,index=cluster_count)
    rPF2_br.to_csv("runningtime_source_destination_fixed_br_"+str(start)+"_"+str(end)+"_run"+".csv")

    rPF2_gnn = pd.DataFrame(times1_gnn,index=cluster_count)
    rPF2_gnn.to_csv("runningtime_source_destination_fixed_gnn_"+str(start)+"_"+str(end)+"_run"+".csv")
    
    cnt = cnt + 1

