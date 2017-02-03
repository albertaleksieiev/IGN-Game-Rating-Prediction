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
 Loss(step 34k-58k):
 ![loss]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss_1.png)
Loss(step 70k-145k):
 ![loss]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss_70-145.png)
Loss(step 70k-200k):
 ![loss]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss_70-200.png)
 ![loss]( https://github.com/AlfredNeverKog/IGN-Game-Rating-Prediction/raw/master/images/loss_70-200-big.png)



 
 - Training Step: 38334  | total loss: 1.06394
 - Training Step: 144524  | total loss: 0.73067
 - Training Step: 200068  | total loss: 0.52506
