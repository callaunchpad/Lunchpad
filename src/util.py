## basically take in two parameters 1. a set of the latent space vectors, 2. a set of the current ingredient cluster
## what we want is the closest other vector in the latent space not in the current ingredient cluster to the cluster
## idea is to create a vector space of the current ingredient cluster and then we find the orthonormal basis of this
## this will be found through gram-schmidt 
## then we run a projection of this and then output the projection with the highest value
## precondition: python/numpy arrays
## post condition vector with the higherst match
## normalization will be taken care of by this function. 

def find_closest(dict_ingredients, my_cluster):
    
    ##expecting in dict_ingredients as a list of tuples
    ##(a, b) where a is the string corresponding to the name of the ingredient
    ## b is the tesnor field for the ingredient
    
    ##NOTE:::: if i get column vectors just transpose
    
    ##convert the tensor into numpy array that are normalized 
    for tup in dict_ingredients:
        tup[1] = tup[1].numpy()
        tup[1] = np.linalg.norm(tup[1]) ##hopefully this doesnt mess some stuff up!!!
    
    size = len(dict_ingredients)
    for i in range(len(dict_ingredients)):
        ##remove the duplicates in my_cluster
        ##kinda slow but wtv!!1
        
        for tup in my_cluster:
            if dict_ingredients[size - 1 - i][0] == tup[0]: ##equality on ingredients
                dict_ingredients.pop(size - 1 - i)
                break ##in case there is some troll that happens
    ##concatenate the my_cluster vectors a, b, c, ... 
    ## normalize my_cluster
    for tup in my_cluster:
        tup[1] = tup[1].numpy()
        tup[1] = np.linalg.norm(tup[1])
    
    ##ASSUMING THE INPUT IS ROW VECTORS IF NOT ROW DO IT
    
    start = my_cluster[0][1]
    
    for i in range(1, len(my_cluster)):
        start = np.concatenate((start, my_cluster[i][1]), axis = 0)
    
    A = np.transpose(start)
    Q = np.zeros(A.shape) ##does not do the edge case of linearly dependent shit ##TODO!!
    for k in range(A.shape[1]):
        avec = A[:, k]
        q = avec
        for j in range(k):
            q = q - np.dot(avec, Q[:,j])*Q[:,j]
    
    Q[:, k] = q/np.linalg.norm(q) 
    
    ##find the dot product sum
    Q = Q.T ##columns are rows
    running = float('-inf')
    best_tuple = (float('-inf'), float('-inf'))
    
    for tup in dict_ingredients:
        vec = tup[1]
        vec = vec.T
        a = 0
        
        for i in range(Q.shape[0]):
            a = a + np.dot(vec, Q[i])
        a = a / Q.shape[0]
        
        if (a > running):
            best_tuple = tup
            running = a
            
    ##this is if we need to find furthest could just update paramaters or make a whole new function forgot we had to do this highkey
    running = float('inf')
    best_tup = (,)
    
    for tup in dict_ingredients:
        vec = tup[1]
        vec = vec.T
        a = 0
        
        for i in range(Q.shape[0]):
            a = a + np.dot(vec, Q[i])
        a = a / Q.shape[0]
        
        if (a < running):
            best_tup = tup
            running = a
    
    return best_tuple ##best_tup for furthest again read above for functional changes 
