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

Model [link](https://github.com/ahmedatef1610/Image-Classification-Web-App-in-Flask-using-Machine-Learning/blob/master/2-dataprepare-ML-pipeline/MLPClassifierModel_model_best.pickle) info:

![ScreenShot_20210802021923](https://user-images.githubusercontent.com/39852784/127789900-0f7de47d-8ee5-476c-a424-f14ec530b41d.png)
![ScreenShot_20210802021904](https://user-images.githubusercontent.com/39852784/127789935-b68c2e50-f741-467d-8470-3f2434923fa5.png)


---

Data [link](https://drive.google.com/file/d/1ARuq78xZCFgUgxCjuAdTKxyv96S4-3wP/view?usp=sharing)

img_all_arrs shape => (2057, 80, 80, 3)

Counter({'bear': 101,
         'cat': 159,
         'chicken': 100,
         'cow': 103,
         'deer': 103,
         'dog': 132,
         'duck': 103,
         'eagle': 100,
         'elephant': 100,
         'human': 100,
         'lion': 102,
         'monkey': 100,
         'mouse': 100,
         'natural': 8,
         'panda': 118,
         'pigeon': 115,
         'rabbit': 100,
         'sheep': 100,
         'tiger': 113,
         'wolf': 100})

![output](https://user-images.githubusercontent.com/39852784/127790994-7f834d77-6f2d-490a-aae0-a6570ed46173.png)

![ScreenShot_20210802023924](https://user-images.githubusercontent.com/39852784/127790569-9cfe1528-5e2b-47c9-97ca-5205ad055096.png)

And we make [data_animals_head_20.pickle](https://github.com/ahmedatef1610/Image-Classification-Web-App-in-Flask-using-Machine-Learning/blob/master/2-dataprepare-ML-pipeline/data_animals_head_20.pickle) contains 

```
data = dict()
data['description'] = 'There are 20 classes and 2057 images are there. All the images are 80 x 80 (rgb)'
data['data'] = img_all_arrs
data['target'] = labels
data['labels'] = set(labels)
```

---

We use scikit-image Image processing library to apply HOG (Histogram of Oriented Gradients) to make feature descriptor that is often used to extract features from image data and use this feature descriptor in train model 

![ScreenShot_20210802024240](https://user-images.githubusercontent.com/39852784/127790673-d2601854-3a41-4f0c-8845-4478d76f6a85.png)

![output](https://user-images.githubusercontent.com/39852784/127790646-519afcc0-4c6c-4bcd-87ad-dfaff1020d3f.png)


---

model evaluation : 
 
![ScreenShot_20210802021739](https://user-images.githubusercontent.com/39852784/127789875-30d4541b-76cf-4c05-b09c-116cc4506aba.png)

---

we use pipeline, BaseEstimator and TransformerMixIn to build our data preprocessing stages

we use GridSearchCV to Hyperparameter Tuning

---


After that we create web app in Flask by rendering HTML, CSS, Boostrap. Then, we finally deploy web app in [Python Anywhere](https://www.pythonanywhere.com/) which is cloud platform. [(App link)](http://ahmedatef1610.pythonanywhere.com/)


---

