from sklearn.ensemble import IsolationForest # Outliers 

x = [[.3], [2], [4], [.5], [90]]

clf = IsolationForest(random_state=0).fit(x)
print(clf.predict(x))
