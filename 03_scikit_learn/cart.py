import numpy as np

def sum_squared_error(y_true, y_pred):
    if len(y_true) > 0:
        return np.square(y_true - y_pred).sum()
    else:
        return 0

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class CART:
    def __init__(self, feature_cols, y_col, min_leaf=20, max_depth=3):
        self.feature_cols = feature_cols
        self.y_col = y_col
        self.min_leaf = min_leaf
        self.max_depth = max_depth

    def get_thresh_per_var(self, df, var):
        var_vals = df[var].unique()
        thresh = np.sort(var_vals)
        sse = np.zeros_like(thresh)
        l_val = np.zeros_like(thresh)
        r_val = np.zeros_like(thresh)
        for i, t in enumerate(thresh):
            idx = df[var] <= t
            l_val[i] = df.loc[idx, self.y_col].mean()
            sse_l = sum_squared_error(df.loc[idx, self.y_col], l_val[i])
            r_val[i] = df.loc[~idx, self.y_col].mean()
            sse_r = sum_squared_error(df.loc[~idx, self.y_col], r_val[i])
            sse[i] = sse_l + sse_r
        
        idx = sse.argmin()
        return thresh[idx], sse.min(), l_val[idx], r_val[idx]

    def get_thresh(self, df):
        best_sse = float("inf")
        for col in self.feature_cols:
            t, sse, l_val, r_val = self.get_thresh_per_var(df, col)
            if sse < best_sse:
                best_sse = sse
                best_col = col
                best_thresh = t
                best_l_val = l_val
                best_r_val = r_val
                
        return best_col, best_thresh, best_l_val, best_r_val

    def fit(self, df, depth=1):
        if len(df) < self.min_leaf or depth > self.max_depth:
            return None
        best_col, thresh, l_val, r_val = self.get_thresh(df)

        node = Node((best_col, thresh, l_val, r_val))

        idx = df[best_col] <= thresh
        node.left = self.fit(df[idx], depth+1)
        node.right = self.fit(df[~idx], depth+1)
        
        return node

def predict(row, tree):
    col, thresh, l_val, r_val = tree.val
    if row[col] <= thresh:
        ans = l_val
        if tree.left:
            return predict(row, tree.left)
        else:
            return ans
    else:
        ans = r_val
        if tree.right:
            return predict(row, tree.right)
        else:
            return ans