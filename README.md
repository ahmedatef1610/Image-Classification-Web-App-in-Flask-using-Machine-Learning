# Image-Classification-Web-App-in-Flask-using-Machine-Learning [(App link)](http://ahmedatef1610.pythonanywhere.com/)


![image](https://user-images.githubusercontent.com/39852784/127789462-24a84c66-0010-4402-8e69-97dfc8753393.png)

--- 

In this project we make neural network model [(MLPClassifierModel)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) to can classifier between    

| index | label |
| --- | --- |
| 0 | wolf |
| 1 | elephant |
| 2 | human |
| 3 | dog |
| 4 | cat |
| 5 | cow |
| 6 | eagle |
| 7 | panda |
| 8 | natural |
| 9 | deer |
| 10 | tiger |
| 11 | rabbit |
| 12 | sheep |
| 13 | chicken |
| 14 | duck |
| 15 | bear |
| 16 | monkey |
| 17 | lion |
| 18 | mouse |
| 19 | pigeon |

---

Model info:

![ScreenShot_20210802021923](https://user-images.githubusercontent.com/39852784/127789900-0f7de47d-8ee5-476c-a424-f14ec530b41d.png)
![ScreenShot_20210802021904](https://user-images.githubusercontent.com/39852784/127789935-b68c2e50-f741-467d-8470-3f2434923fa5.png)


---

We use scikit-image Image processing library to apply HOG (Histogram of Oriented Gradients) to make feature descriptor that is often used to extract features from image data and use this feature descriptor in train model and model evaluation : 

![ScreenShot_20210802021739](https://user-images.githubusercontent.com/39852784/127789875-30d4541b-76cf-4c05-b09c-116cc4506aba.png)

---

we use pipeline, BaseEstimator and TransformerMixIn to build our data preprocessing stages

we use GridSearchCV to Hyperparameter Tuning

---


After that we create web app in Flask by rendering HTML, CSS, Boostrap. Then, we finally deploy web app in [Python Anywhere](https://www.pythonanywhere.com/) which is cloud platform. [(App link)](http://ahmedatef1610.pythonanywhere.com/)


---

