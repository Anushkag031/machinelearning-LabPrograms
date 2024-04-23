from numpy.lib.function_base import average


def evalclass(x_test, y_test, model):
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,ConfusionMatrixDisplay

  y_pred=model.predict(x_test)
  
  accuracy=accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  f1=f1_score(y_test, y_pred,average="weighted")
  print("F1:", f1)
  confusion_matrix=confusion_matrix(y_test, y_pred,labels=[0,1,2,3])
  
  display=ConfusionMatrixDisplay(confusion_matrix,display_labels=[0,1,2,3])
  display.plot()
  print(classification_report(y_test, y_pred))



