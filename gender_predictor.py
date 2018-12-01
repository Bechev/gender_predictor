from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#height, weight
X = [[190,90],[175,65],[160,50],[175,57],[170,100],[180,80],[162,45],[160,51],[165,49],[179,75]]
#male = 0 ; female = 1
Y = [0,0,1,1,1,1,0,1,1,0]

clf_classifier = tree.DecisionTreeClassifier()
clf_classifier = clf_classifier.fit(X,Y)
prediction_classifier =  clf_classifier.predict([[174,61]])
print("DecisionTreeClassifier: ")
print(prediction_classifier)

clf_regressor = tree.DecisionTreeRegressor()
clf_regressor = clf_regressor.fit(X,Y)
prediction_regressor =  clf_regressor.predict([[174,61]])
print("DecisionTreeRegressor: ")
print(prediction_regressor)

clf_rnd_forest = RandomForestClassifier(n_estimators=10)
clf_rnd_forest = clf_rnd_forest.fit(X,Y)
prediction_rnd_forest =  clf_rnd_forest.predict([[174,61]])
print("RandomForestClassifier: ")
print(prediction_rnd_forest)