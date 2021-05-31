from sklearn.svm import SVC
import numpy as np
import numexpr as ne


class MSVC(SVC):
    
    
    def __init__(self, width, height, C=1.0, kernel='rbf', gamma='scale', 
                 beta = None, neigh_type = 0, deep = 1, bias = 0,
                 degree=3, coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, break_ties=False,
                 random_state=None):
        
        self.neigh_type = neigh_type
        self.deep = deep
        self.bias = bias
        self.mrbf = False
        self.beta = beta
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.break_ties = break_ties
        self.random_state = random_state
        self.width = width
        self.height = height
        
        if kernel == "mrbf":
            self.mrbf = True
        

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        if self.beta == None:
            self.beta = [0 for i in range(int(len(self.classes)*(len(self.classes)-1)/2))]
        if self.gamma == 'scale':
            self.gamma = 1/(X.var()*X.shape[1])
        elif self.gamma == 'auto':
            self.gamma = 1/(X.shape[1])
        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):
                X_1, y_1, index = self.div_cl(X, y, fir_cl = self.classes[i], sec_cl = self.classes[j])
                
                if self.mrbf:
                    self.kernel = self.my_kernel(self.gamma, self.beta[int((2*len(self.classes)-1-i)*i/2)+j-i-1], index = X.shape[1])
                
                model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, 
                 degree=self.degree, coef0=self.coef0, shrinking=self.shrinking, probability=self.probability,
                 tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight,
                 verbose=self.verbose, max_iter=self.max_iter, 
                 break_ties=self.break_ties, random_state=self.random_state)
                
                if self.mrbf:
                    self.models.append(model.fit(self.merge(X_1,self.neighbor_labels(y, width = self.width, height = self.height, cl1 = self.classes[i], cl2 = self.classes[j], neigh_type = self.neigh_type,
                    deep = self.deep, bias = self.bias)[index]), y_1))
                else:
                    self.models.append(model.fit(X_1, y_1))

                self.beta[int((2*len(self.classes)-1-i)*i/2)+j-i-1] = self.beta_search(
                    self.models[int((2*len(self.classes)-1-i)*i/2)+j-i-1], X_1, y, index = index, gamma = self.gamma, width = self.width, height = self.height,
                    neigh_type = self.neigh_type, deep = self.deep, bias = self.bias, fir_cl = self.classes[i], sec_cl = self.classes[j])
        
        return self
    
    def predict(self, X, width, height, y = None):
        y_pred = [] 
        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):
                if self.mrbf:
                    y_pred.append(self.models[int((2*len(self.classes)-1-i)*i/2)+j-i-1].predict(self.merge(X,self.neighbor_labels(y, width, height, cl1 = self.classes[i], cl2 = self.classes[j], neigh_type = self.neigh_type,
                    deep = self.deep, bias = self.bias))))
                else:
                    y_pred.append(self.models[int((2*len(self.classes)-1-i)*i/2)+j-i-1].predict(X))
        
        y_ans = np.zeros([len(y_pred[0])])
        for i in range(len(y_ans)):
            count = [0 for k in range(max(self.classes)+1)]
            for j in range(int(len(self.classes)*(len(self.classes)-1)/2)):
                count[int(y_pred[j][i])] += 1
            for k in range(len(count)):
                if count[k] == max(count):
                    y_ans[i] = k
                    break
                    
        return y_ans
    
    @staticmethod
    def beta_search(self, x, y, index, gamma, width, height, neigh_type = 0, deep = 1, bias = 0, fir_cl = 0, sec_cl = 1):
        
        alpha = self.dual_coef_
        b = self.intercept_
        supp = self.support_

        eps = MSVC.neighbor_labels(y, width, height, cl1 = fir_cl, cl2 = sec_cl, neigh_type = neigh_type,
            deep = deep, bias = bias)[index]
        y = y[index]
        mx = 0
        mn = 0
        xj = x[supp]
        E10 = alpha
        E20 = np.sum(alpha*eps[supp])
        for i in range(len(y)):
            if y[i] == fir_cl:
                yi = 1
            else:
                yi = -1
            E1 = (np.sum(E10*(np.sum(np.exp(-gamma*(x[i]-xj)**2),axis = 1)))+b)*yi
            E2 = E20*yi*eps[i]
            if E2 > 0 and -E1/E2 > mx:
                if mn >= -E1/E2:
                    mx = -E1/E2
            elif E2 < 0 and -E1/E2 < mn:
                if -E1/E2 >= mx:
                    mn = -E1/E2
            elif E2 < 0 and mn == 0:
                mn = -E1/E2
        return mn
    
    
    @staticmethod
    def my_kernel(gamma = 1, beta = 0, index = 1):
        def MRF_kernel(X1, X2):
            return ne.evaluate('exp(-gamma*(A+B-2*C)) + beta * D', {
                    'A' : np.einsum('ij,ij->i',X1[:,:index],X1[:,:index])[:,None],
                    'B' : np.einsum('ij,ij->i',X2[:,:index],X2[:,:index])[None,:],
                    'C' : np.dot(X1[:,:index], X2[:,:index].T),
                    'D' : (X1[:,index]*X2[:,index,None]).T,
                    'gamma' : gamma,
                    'beta' : beta
                   })
        return MRF_kernel
        
    @staticmethod
    def neigh_index(i, width, height, neigh_type, deep = 1, bias = 0):
        neighs = []
        if neigh_type == 0:
            if (i % width) > 0:
                neighs.append(i-1)
            if (i % width) < width-1:
                neighs.append(i+1)
            if (i % (width*height)) >= width:
                neighs.append(i-width)
            if (i % (width*height)) < (height-1)*width:
                neighs.append(i+width)
        elif neigh_type == 1:
            for j in range(2*deep+1):
                for l in range(2*deep+1):
                    if (i % width) - (deep-l)  < 0 and l < deep:
                        continue
                    if (i % width) + (l-deep) >= width and l > deep:
                        continue
                    if (i % (width*height)) - (deep-j)*width < 0 and j < deep:
                        continue
                    if (i % (width*height)) + (j-deep)*width >= height*width and j > deep:
                        continue    
                    n = i - deep - width*deep + l + j*width 
                    if n != i:
                        neighs.append(n)
        elif neigh_type == 2:
            for j in range(2*deep+1):
                for l in range(2*deep+1):
                    if (i % width) - (deep-l)  < 0 and l < deep:
                        continue
                    if (i % width) + (l-deep) >= width and l > deep:
                        continue
                    if (i % (width*height)) - (deep-j)*width < 0 and j < deep:
                        continue
                    if (i % (width*height)) + (j-deep)*width >= height*width and j > deep:
                        continue    
                    n = i - deep - width*deep + l + j*width 
                    if n != i:
                        neighs.append(n)
            for j in range(2*bias+1):
                if (i % width) - (bias-j)  >= 0 and j < bias:
                        continue
                if (i % width) + (j-bias) >= width and j > bias:
                        continue
                if (i % (width*height)) - bias*width >= 0:
                    neighs.append(i - bias - width*bias + j)
                if (i % (width*height)) +bias*width < height*width:
                    neighs.append(i - bias + width*bias + j)
            if (i % width) - bias >= 0:
                neighs.append(i - bias)
            if (i % width) + bias < width:
                neighs.append(i + bias)
        return neighs

    @staticmethod
    def neighbor_labels(Y, width, height, cl1 = 0, cl2 = 1, neigh_type = 0, deep = 1, bias = 0):
        if not (Y is None):
          Y2 = np.zeros([len(Y)])
          for i in range(len(Y)):
              count_0 = 0
              count_1 = 0
              for j in MSVC.neigh_index(i, width, height, neigh_type, deep, bias):
                  if Y[j] == cl2:
                      count_0 += 1
                  elif Y[j] == cl1:
                      count_1 += 1
              Y2[i] = count_1 - count_0
          return Y2
        else:
          return []
      
    @staticmethod
    def div_cl(X, y, fir_cl, sec_cl):
        n = ((y == fir_cl) | (y == sec_cl)).sum()
        X_new = np.zeros([n, X.shape[1]])
        y_new = np.zeros([n])
        index = np.zeros([n], dtype = int)
        j = 0
        for i in range(len(y)):
            if y[i] == fir_cl or y[i] == sec_cl:
                X_new[j] = X[i]
                y_new[j] = y[i]
                index[j] = i
                j += 1
        return X_new, y_new, index
    
    @staticmethod
    def merge(X, y):
        return np.append(X, y[:,None], axis = 1)