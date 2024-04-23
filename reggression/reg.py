def eval(y_test,y_pred):
  import pandas as pd
  from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
  import numpy as np

model=pd.DataFrame({'actual val':y_test,'pred val':y_pred})
print(model)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
np.sqrt(mse)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print('MAE:',mae)
print('MSE:',mse)
print('RMSE:',rmse)
print('R2:',r2)