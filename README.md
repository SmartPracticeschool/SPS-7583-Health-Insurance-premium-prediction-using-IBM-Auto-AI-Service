# SPS-7583-Health-Insurance-premium-prediction-using-IBM-Auto-AI-Service
Health Insurance-premium-prediction  using IBM Auto AI Service

# Create an application to predict your insurance premium cost with AutoAI 

![finalDemo](https://user-images.githubusercontent.com/10428517/82013347-f7e71500-962e-11ea-9c28-2dec7d5b30cd.gif)

As shown above, this application leverages machine learning models to predict your insurance charges, and helps the customer understand how smoking or decreasing your BMI affects
insurance premiums.

As we see the value of gross insurance premiums worldwide continue to skyrocket past 5 trillion dollars,
we know that most of these costs are preventable. For example, just by eliminating smoking, and lowering
your BMI by a few points could mean shaving thousands of dollars off of your premium charges. In this 
application, we study the effects of age, smoking, BMI, gender, and region to determine how much of 
a difference these factors can make on your insurance premium.  By using our 
application, customers see the radical difference their lifestyle choices make on their insurance 
charges. By leveraging AI and machine learning, we help customers understand just how much smoking increases their premium, by predicting how much they will have to pay within seconds.

## Description

Using IBM AutoAI, you automate all the tasks involved in building predictive models for different requirements. You see how AutoAI generates great models quickly which save time and effort and aid in faster decision-making process. You create a model that from a data set that includes the age, sex, BMI, number-of-children, smoking preferences, region and charges to predict the health insurance premium cost that an individual pays.

When you have completed this code pattern, you understand how to:

* Setup, quickly, the services on IBM Cloud for building the model.
* Ingest the data and initiate the AutoAI process.
* Build different models using AutoAI and evaluate the performance.
* Choose the best model and complete the deployment.
* Generate predictions using the deployed model by making REST calls.
* Compare the process of using AutoAI and building the model manually.
* Visualize the deployed model using a front-end application.

### Architecture Components

![Architecture Components](https://media.github.ibm.com/user/21063/files/3b77e580-913c-11ea-9dea-425b1d4f4ee0)

## Flow Description
1. The user creates an IBM Watson Studio Service on IBM Cloud.
2. The user creates an IBM Cloud Object Storage Service and adds that to Watson Studio.
3. The user uploads the insurance premium data file into Watson Studio.
4. The user creates an AutoAI Experiment to predict insurance premium on Watson Studio
5. AutoAI uses Watson Machine Learning to create several models, and the user deploys the best performing model.
6. The user uses the Flask web-application to connect to the deployed model and predict an insurance charge.

## Included components
*	[IBM Watson Studio](https://cloud.ibm.com/catalog/services/watson-studio) - IBM Watson® Studio helps data scientists and analysts prepare data and build models at scale across any cloud.
*	[IBM Watson Machine Learning](https://cloud.ibm.com/catalog/services/machine-learning) - IBM Watson® Machine Learning helps data scientists and developers accelerate AI and machine-learning deployment. 
*	[IBM Cloud Object Storage](https://cloud.ibm.com/catalog/services/cloud-object-storage) - IBM Cloud™ Object Storage makes it possible to store practically limitless amounts of data, simply and cost effectively.

## Featured technologies
+ [artificial-intelligence](https://developer.ibm.com/technologies/artificial-intelligence/) - Build and train models, and create apps, with a trusted AI-infused platform.
+ [Python](https://www.python.org/) - Python is an interpreted, high-level, general-purpose programming language.

## Watch the Video

#### IBM Watson AutoAI Part 1/3: Data exploration and visualization

<a href="https://www.youtube.com/watch?v=9JuiqVXvQ74">
<img src="https://user-images.githubusercontent.com/10428517/82686741-2c4c6980-9c0b-11ea-896d-b8972aac981a.png" width="725" height="388" /> 
</a>

#### IBM Watson AutoAI Part 2/3: Running AutoAI

<a href="https://www.youtube.com/watch?v=ilw6O5HwtY0">

<img src="https://user-images.githubusercontent.com/10428517/82686738-2bb3d300-9c0b-11ea-9987-67f40951aa81.png" width="725" height="388" />
</a>

#### IBM Watson AutoAI Part 3/3: Connecting model API to a web-app

<a href="https://www.youtube.com/watch?v=sOtezE-YNPU">

<img src="https://user-images.githubusercontent.com/10428517/82686732-2787b580-9c0b-11ea-99cb-2986cacead71.png" width="725" height="388" />
</a>

## Prerequisites

This Cloud pattern assumes you have an **IBM Cloud** account. Go to the 
link below to sign up for a no-charge trial account - no credit card required. 
  - [IBM Cloud account](https://tinyurl.com/y4mzxow5)
  - [Python 3.8.2](https://www.python.org/downloads/release/python-382/)

# Steps
0. [Download the data set ](#step-0-Download-the-data-set)
1. [Clone the repo](#step-1-clone-the-repo)
2. [Explore the data (optional)](#step-2-explore-the-data-optional)
3. [Create IBM Cloud services](#step-3-create-ibm-cloud-services)
4. [Create and Run AutoAI experiment](#step-4-create-and-run-autoai-experiment)
5. [Create a deployment and test your model](#step-5-create-a-deployment-and-test-your-model)
6. [Create a notebook from your model (optional)](#step-6-create-a-notebook-from-your-model-optional)
7. [Run the application](#step-7-run-the-application)

## Step 0. Download the data set 
We will use an insurance data set from Kaggle. You can find it [here](https://www.kaggle.com/noordeen/insurance-premium-prediction).
 Click on the `Download` button, and you should see
that you will download a file named `insurance-premium-prediction.zip`. Once you unzip the file, you should see `insurance.csv`.
This is the data set we will use for the remainder of the example. Remember that this example is purely educational, and you
could use any data set you want - we just happened to choose this one.

## Step 1. Clone the repo
Clone this repo onto your computer in the destination of your choice:
```
git clone https://github.com/IBM/predict-insurance-charges-with-ai
```
This gives you access to the notebooks in the `notebooks` directory. To explore the data before creating a model, 
you can look at the [Claim Amount Exploratory](https://github.com/IBM/predict-insurance-charges-with-ai/blob/master/notebooks/Claim%20Amount%20Exploratory.ipynb) notebook, and create a [IBM Cloud Object Storage](https://cloud.ibm.com/catalog/services/cloud-object-storage) service, and paste your credentials in the notebook to run it. This step is purely optional.

## Step 2. Explore the data (optional)

#### If you want to run the notebook that is explored below, go to [`notebooks/Claim Amount Exploratory.ipynb`](https://github.com/IBM/predict-insurance-charges-with-ai/blob/master/notebooks/Claim%20Amount%20Exploratory.ipynb).
* Within Watson Studio, you explore the data before you create any 
machine learning models. You want to understand the data, and find any trends between 
what you are trying to predict (insurance premiums <b>charges</b>) and the data's features.

* Once you import, you see the data into a data frame, and call the 
`df_claim.head()` function, you see the first 5 rows of the data set. 
You see the features to be `age`, `sex`, `bmi`, `children`, `smoker`,
and `region`.

![scatter](https://media.github.ibm.com/user/79254/files/ed325a80-8a48-11ea-8fcf-d1e9877458ef)

* To check if there is a strong relationship between `bmi` and `charges` you 
create a scatter plot using the seaborn and matplotlib libraries. You 
see that there is no strong correlation between `bmi` and `charges`,
as shown below.

![scatter](https://media.github.ibm.com/user/79254/files/2965bb00-8a49-11ea-81f9-a528fc1e2606)

* To check if there is a strong relationship between `sex` and `charges` you create a box plot. You see that the average claims for males and females are similar, whereas males have a bigger proportion of the higher claims.

![scatter](https://media.github.ibm.com/user/79254/files/32ef2300-8a49-11ea-93aa-990f85eccf9d)

* To check if there is a strong relationship between being a `smoker` and `charges` you create a box plot. You see that if you are a smoker, your claims are much higher on average.

![scatter](https://media.github.ibm.com/user/79254/files/4221a100-8a48-11ea-8104-64f50d8ae92f)

* Let's see if the `smoker` group is well represented. As you see, below, it is. 
There are around 300 smokers, and around 1000 non-smokers.

![scatter](https://media.github.ibm.com/user/79254/files/477eeb80-8a48-11ea-83a0-9a073bf4f176)

* To check if there is a strong relationship between being a `age` and `charges` you create a scatter plot. You see that claim amounts increase with age, and tend to form groups around 12,000, 30,000, and 40,000.

![scatter](https://media.github.ibm.com/user/79254/files/5bc2e880-8a48-11ea-8dad-8effab71a8ac)

If you want to see all of the code, and run the notebook yourself, check the data folder above.

## Step 3. Create IBM Cloud services

First login to your IBM Cloud account. Use the video below for directions on how to create IBM Watson Studio Service.

![watsonStudio](https://media.github.ibm.com/user/79254/files/e493eb80-8626-11ea-87b5-f1c7cf8d50e0)

* After logging into IBM Cloud, click `Proceed` to show that you have read your data rights.

* Click on `IBM Cloud` in the top left corner to ensure you are on the home page.

* Within your IBM Cloud account, click on the top search bar to search for cloud services and offerings. Type in `Watson Studio` and then click on `Watson Studio` under `Catalog Results`.

* This takes you to the Watson Studio service page. There you can name the service as you wish. For example, one may name it 
`Watson-Studio-trial`. You can also choose which data center to create your instance in. The gif above shows mine as 
being created in Dallas.

* For this guide, you choose the `Lite` service, which is no-charge. This has limited compute; it is enough
to understand the main functionality of the service.

* Once you are satisfied with your service name, and location, and plan, click on create in the bottom-right corner. This creates your Watson Studio instance. 

![createProj](https://user-images.githubusercontent.com/10428517/81858932-5fab3c00-9519-11ea-9301-3f55d9e2e98d.gif)

* To launch your Watson Studio service, go back to the home page by clicking on `IBM Cloud` in the top-left corner. There you see your services, and under there you should see your service name. This might take a minute or two to update. 

* Once you see your service that you just created, click on your service name, and this takes you to your 
Watson Studio instance page, which says `Welcome to Watson Studio. Let's get started!`. Click on the `Get Started` button.

* This takes you to the Watson Studio tooling. There you see a heading that says `Start by creating a project` and a button that says `Create Project`. Click on `Create a Project`. Next click on `Create an Empty project`.

* On the create a new project page, name your project. One may name the project - `insurance-demo`. You also need to associate an IBM Cloud Object store instance, so that you store the data set.

* Under `Select Storage service` click on the `Add` button. This takes you to the IBM Cloud Object Store service page. Leave the service on the `Lite` tier and then click the `Create` button at the bottom of the page. You are prompted to name the service and choose the resource group. Once you select a name, click the resource group `Confirm` button. 

* Once you've confirmed your IBM Cloud Object Store instance, you are taken back to the project page. Click on `refresh` and you should see your newly created Cloud Object Store instance under `Storage`. That's it! Now you can click `Create` at the bottom right of the page to create your first IBM Watson Studio project :) 

![addData](https://media.github.ibm.com/user/79254/files/0e054500-8630-11ea-99dc-7e13ce87bd9d)

* Once you have created your Watson Studio Project, you see a blue `Add to Project` button on the top-right corner of your screen. Click on `Add to Project` and then select `Data`. This brings up a column on the right-hand side that says `Data`. 

* In the Data column, click on `browse` to add data from a file. Go into where you downloaded your dataset from 
[Step 0](https://github.com/IBM/predict-insurance-charges-with-autoai#step-0-download-the-data-set) and then navigate
to the `data` folder, and then select `insurance.csv`. 

* Watson Studio takes a couple of seconds to load the data, and then you should see the import has completed. To make sure it has worked properly, you can click on `Assets` on the top of the page, and you should see your 
insurance file under `Data Assets`. 

## Step 4. Create and Run AutoAI experiment

![createAutoAI](https://user-images.githubusercontent.com/10428517/81858928-5de17880-9519-11ea-9da6-4721f5ad601c.gif)

* Once you've created your project, click on the `Add to project` at the top-right of your Watson Studio project page. This  pops up an image with different assets you can choose to add to your project. Click on `AutoAI experiment`.

* This takes you to a page which says `New AutoAI experiment` at the top-left. Name your experiment as you want. One may name it `auto-ai-insurance-demo`.

* Next, you need to add a Watson Machine Learning instance before you create the Watson AutoAI experiment. On the right side of the screen click on `Associate a Machine Learning instance`. 

* Same as before, select the `Lite` Tier, and click on the `Create` button at the bottom of the page. Name your instance as you wish. One may name it named mine `machine-learning-free`. Choose the location and the resource group and then click on `Confirm` when you are happy with your instance details.

* Once you create your machine learning service, you are taken back to the new AutoAI experiment page. Click on 
`Reload` on the right side of the screen. You should see your newly created machine learning instance. Great job! Click on `Create` on the bottom right part of your screen to create your first AutoAI experiment!

![experimentSettings](https://media.github.ibm.com/user/79254/files/05ad0a00-8630-11ea-94e7-cd47ae3ac941)

* After you create your experiment, you are taken to a page to add a data source to your project. Click on `Select from project` and then add the `insurance.csv` file. Click on `Select asset` to confirm your data source.



* Next, you see that AutoAI processes your data, and you see a `What do you want to predict` section. 
Select the `charges` as the `Prediction column`. 

![experimentSettings](https://media.github.ibm.com/user/79254/files/4e63ac00-8fbc-11ea-842d-7107de2fed13)

* Next, let's explore the AutoAI settings to see what you can customize when running your experiment. Click on `Experiment settings.` First, you see the `data source` tab, which lets you omit 
certain columns from your experiment. You choose to leave all columns. You can also select the 
training data split. It defaults to 85% training data. The data source tab also shows which metric you  
optimize for. For the regression, it is RMSE (Root Mean Squared Error), and for other types of experiments,
such as Binary Classification, AutoAI defaults to Accuracy. Either way, you can change the metric from this tab depending on your use case.

* Click on the `Prediction` tab from within the `Experiment settings`. There you can select from Binary Classification, Regression, and Multiclass Classification.

* Lastly, you can see the `Runtime` tab from the `Experiment settings` this shows you other experiment details 
you may want to change depending on your use case. 

* Once you are happy with your settings, ensure you are predicting for the `charges` column, and click on the run `Run Experiment` button on the bottom-right corner of the 
screen.

![compl](https://media.github.ibm.com/user/79254/files/004fbf80-8630-11ea-9c69-e97b12c39bbe)

* Next, your AutoAI experiment runs on its own. You see a progress map on the right side of the screen
which shows which stage of the experiment is running. This may be Hyper Parameter Optimization, feature engineering, 
or some other stage.

* You have different pipelines that are created, and you see the rankings of each model. Each model is ranked based on the metric that you selected. In the specific case that is the RMSE(Root mean squared error). Given that you want that number to be as small as possible, you can see that in the experiment, the model with the smallest RMSE is at the top of the leaderboard.

* Once the experiment is done, you see `Experiment completed` under the Progress map on the right hand side of
the screen. 

![compl](https://media.github.ibm.com/user/79254/files/38963a00-8a44-11ea-9696-377f268b7af6)

* Now that AutoAI has successfully generated eight different models, you can rank the models by different metrics, such as explained variance, root mean squared error, R-Squared, and mean absolute error. Each time you select a different metric, the models are re-ranked by that metric.

* Let's pick RMSE as the experiment's metric. You see the smallest RMSE value is 4514.389, from Pipeline 8. Click on `Pipeline 8`.

* On the left-hand side, you can see different `Model Evaluation Measures`. For this particular model, you can view the metrics, such as explained variance, RMSE, and other metrics.

* On the left-hand side, you can also see `Feature Transformations`, and `Feature Importance`.

* On the left-hand side, click on `Feature Importance`. You can see here that the most important predictor of the insurance premium is whether you are a `smoker` or `not-smoker`. This is by far the most important feature, with `bmi` coming in as the second most important. This makes sense, given that many companies offer discounts for employees who do not smoke.

## Step 5. Create a deployment and test your model
![compl](https://media.github.ibm.com/user/79254/files/4ea8f800-8a4e-11ea-9da5-f87bff6f4fef)

* Once you are ready to deploy one of the models, click on `Save As` at the top-right corner of the model you want to deploy. Save it as a `Model`. You show you how to save it as a notebook in step 6. 

* Name your model as you want, one may name it `Insurance Premium Predictor - Pattern Demo`.

* Once you have finished saving it as a deployment, you see a green notification at the top right of your screen saying that your model has been successfully saved. Click on `View in Project` on that notification at the top-right corner of your screen.

* Next, you are taken to a screen that has the name of the model you just saved. Click on `Deployments` from the Tab in the middle of the screen. 

* Next, click on the `Add Deployment` button on the right-side of the screen. Name your deployment as you want. One may name it `demo-deployment` and then click `Save`.

* On your saved model overview page, you should see your new deployment `demo-deployment` being initialized.

![compl](https://media.github.ibm.com/user/79254/files/caa34000-8a4e-11ea-9142-b1e19a482b94)

* Click on `demo-deployment` or whatever you named your deployment.

* It takes a few minutes for the deployment to be complete. Once it is complete - you see that a `Test` tab appears in the top of the screen. Click on the `Test` tab.

* Here you can test your model. Enter input data such as `age`, `bmi`, `children`, `smoker` and `region`, and then click the `Predict` button at the bottom of the screen.

* As you can see, the model predicted a premium of 4655, when you enter age 27, bmi: 22, children: 0, smoker: no, region: southwest.

* To validate the prediction, you check the data file that you used to train the model, and see
a row that has similar inputs to what was inputted. You can find a male, 26 year old, with 0 children,
non-smoker to get a premium of 3,900. This is relatively close to the model's prediction, so 
we know the model is working properly.



### 6 Get IBM Cloud API key

<!-- ![apikey-instanceID](https://media.github.ibm.com/user/79254/files/4119b680-8e30-11ea-8bc3-97ab1558fc23) -->
* Generate an IBM Cloud apikey by going to `cloud.ibm.com` and then from the top-right part of the screen click on `Manage`-> `IAM`.

![manage](https://user-images.githubusercontent.com/10428517/95252784-4d84af80-07d2-11eb-9fdd-1fe119329cef.png)


* Next, click on `API keys` from the left side-bar. Next click on `Create an IBM Cloud API key`.

![create-api-key](https://user-images.githubusercontent.com/10428517/95252429-d4855800-07d1-11eb-80e3-fd3b55d5d0a8.png)


* Name the key as you wish, and then click `Create`. 

![create](https://user-images.githubusercontent.com/10428517/95252417-d222fe00-07d1-11eb-95a9-8bd7c9d4bbce.png)


* Once the key is created, click on the `Download` button.

![download](https://user-images.githubusercontent.com/10428517/95252393-ccc5b380-07d1-11eb-8d14-9d7154f71b86.png)

### 7 Get model deployment ID

<!-- ![model-deploy-url](https://user-images.githubusercontent.com/10428517/81858555-caa84300-9518-11ea-9088-3f088216da83.gif) -->

* From inside Watson Studio (Or Cloud Pak for Data), click on `Deployment Spaces`. 

* From there, click on the name of the deployment in which you deployed your model to.

* Next, click on on the name of the model.

* Next, click on the deployment of the model.

* From there, you will be taken to the deployment API reference page - on the right hand side you can see the `Deployment ID`. Go ahead and copy that 
and keep it handy - you will need to paste that into your `app.py` page.

![deploy-id](https://user-images.githubusercontent.com/10428517/95250925-a737aa80-07cf-11eb-9ff2-a51399f7c300.png)


### 8 Generate the access token

* From the command line, type ```curl -V``` to verify if cURL is installed in your system. If cURL is not installed, refer to [this](https://develop.zendesk.com/hc/en-us/articles/360001068567-Installing-and-using-cURL#install) instructions to get it installed.

* Execute the following cURL command to generate your access token, but replace the apikey with the 
apikey you got from [step 7.1](https://github.com/IBM/predict-insurance-charges-with-autoai#71-get-IBM-Cloud-API-key) above. 

```
curl -X POST 'https://iam.cloud.ibm.com/oidc/token' -H 'Content-Type: application/x-www-form-urlencoded' -d 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=<api-key-goes-here>'
```

As shown in the image below, the apikey can be copy and pasted from the downloaded file from the end of [step 7.1](https://github.com/IBM/predict-insurance-charges-with-autoai#71-get-IBM-Cloud-API-key). The curl request would look something like this after the apikey is pasted in:

![api](https://user-images.githubusercontent.com/10428517/95252350-c0d9f180-07d1-11eb-841e-d5cd72da72d4.png)

```
curl -X POST 'https://iam.cloud.ibm.com/oidc/token' -H 'Content-Type: application/x-www-form-urlencoded' -d 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=aSULp7nFTJl-jGx*******aQXfA6dxMlpuQ9QsOW'
```

[app-Url](https://node-red-jsnym-2020-11-01.eu-gb.mybluemix.net/ui/)
