# IGN-Game-Rating-Prediction
Trying to predict average game rating. Use column ['score_phrase'] in sentemental layer, ['platform', 'genre', 'editors_choice', 'release_year'] - as categorical properties. 
# Chalenge 
Build NN with more than one inputs, and more than one layers to merge.
# Dependencies
 - pandas
 - tflearn
 - tensorflow
 - numpy
 - sklearn
 
#Result
 Achitecture: 
 ![achitecture]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/architecture.png)
 Loss:
 ![loss]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss.png)
 Loss(step 34k-38k):
 ![loss_next]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss_1.png)

 
 ##### Training Step: 38334  | total loss: 1.06394
