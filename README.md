# Beyond The Buzz
## Author

- [rock42069](https://www.github.com/rock42069)

## Data Preprocessing

Given the file 'train.csv',I checked the file on excel sheet using histograms to get an overview about outliers present in the data.
To filter the data we used z-score as a parameter.
Whenever the z-score of a datapoint was either greater than 3 or less than -3, we scraped the data. Here, train is pandas dataframe and stats.zscore is imported from scipy library.

```python
from scipy import stats
for i in params:
  train = train[abs(stats.zscore(train[i]))<=3]
```


Raw Data | Processed Data
--- | ---
![Alt text](https://i.ibb.co/St5fH13/newplot.jpg) | ![Alt text](https://i.ibb.co/gyXwzCy/newplot-3.jpg)
![Alt text](https://i.ibb.co/fvqWqQ1/newplot-1.jpg) | ![Alt text](https://i.ibb.co/Rv20CkT/newplot-4.jpg)

 




Before normalization, the train dataframe (top 5 rows) looked like - 

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|39353|85475|117961|118300|123472|117905|117906|290919|117908|
|1|1|17183|1540|117961|118343|123125|118536|118536|308574|118539|
|2|1|36724|14457|118219|118220|117884|117879|267952|19721|117880|
|3|1|36135|5396|117961|118343|119993|118321|240983|290919|118322|
|4|1|42680|5905|117929|117930|119569|119323|123932|19793|119325|


Finally,I normalisized the data so that it has a mean of 0 and a standard deviation of 1 so that scale of the data is uniform. Here, trainprocessed is the filtered dataset.

```python
numerical_cols = ['PARAMETER_1', 'PARAMETER_2', 'PARAMETER_3', 'PARAMETER_4', 'PARAMETER_5', 'PARAMETER_6', 'PARAMETER_7', 'PARAMETER_8', 'PARAMETER_9']
scaler = StandardScaler()

train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
```

 Final Data - 

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|-0\.06970642898763442|2\.2884185393973593|0\.09751489082811729|-0\.11849269255364434|0\.8139514260170707|-0\.3081528224112156|-0\.7225140110922403|1\.0456337456753682|-0\.6103052796765416|
|1|1|-0\.8849008589800971|-0\.799671007066648|0\.09751489082811729|-0\.0706434921686976|0\.7336061551539824|-0\.2159706704018587|-0\.7132748606465639|1\.2206836867401167|-0\.3318021068751961|
|2|1|-0\.16637517758710146|-0\.3244359719600109|0\.1605620090515678|-0\.20751446071168478|-0\.4799084402161773|-0\.3119511361231701|1\.4779583187045469|-1\.6433038876793558|-0\.6226635821146361|
|3|1|-0\.18803280136994135|-0\.6578032130454982|0\.09751489082811729|-0\.0706434921686976|0\.008414834280229261|-0\.24737980301994386|1\.082449353038754|1\.0456337456753682|-0\.42757895077042907|
|4|1|0\.052627890750070724|-0\.6390763699031835|0\.08969509321900714|-0\.5302183702845814|-0\.08975921426717537|-0\.10099863612077489|-0\.6341408038134363|-1\.6425900051669422|0\.01511309727988711|

Then finally for selecting the parameters, I've checked the correlation between all parameters, from the table obtained it was quite obvious that there is no correlation between them.

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|VERDICT|1\.0|0\.007374874140603658|-0\.019030255915717997|-0\.014258449027845751|-0\.020536733558475156|-0\.009925964011483827|0\.006333060608162153|0\.009890192410095809|0\.003442055686773596|0\.010322844346048021|
|PARAMETER\_1|0\.007374874140603658|1\.0|0\.010860591634781303|-0\.057794658984812536|-0\.007577121962256353|0\.0009147055282313084|-0\.02347419475193103|0\.009076361819325739|0\.05645086639110787|0\.004495343750986927|
|PARAMETER\_2|-0\.019030255915717997|0\.010860591634781303|1\.0|-0\.07671399120182355|0\.031834285925097724|-0\.06672360673094148|0\.040485155310006404|-0\.045797874184217886|-0\.1550985266629161|-0\.025561490682054946|
|PARAMETER\_3|-0\.014258449027845751|-0\.057794658984812536|-0\.07671399120182355|1\.0|0\.23690188383288915|0\.03216095051930841|0\.01524605135333103|0\.07514775886987143|-0\.04533840533278488|0\.0017472577315428624|
|PARAMETER\_4|-0\.020536733558475156|-0\.007577121962256353|0\.031834285925097724|0\.23690188383288915|1\.0|0\.0755539000007956|-0\.011615159191281978|0\.03935912107958597|0\.06898399553363642|0\.026635553646687377|
|PARAMETER\_5|-0\.009925964011483827|0\.0009147055282313084|-0\.06672360673094148|0\.03216095051930841|0\.0755539000007956|1\.0|-0\.013527466272678269|0\.06838376113331358|0\.0842034527574188|0\.06667492174678759|
|PARAMETER\_6|0\.006333060608162153|-0\.02347419475193103|0\.040485155310006404|0\.01524605135333103|-0\.011615159191281978|-0\.013527466272678269|1\.0|0\.039345835316920576|-0\.14520089333558753|0\.2700329859552825|
|PARAMETER\_7|0\.009890192410095809|0\.009076361819325739|-0\.045797874184217886|0\.07514775886987143|0\.03935912107958597|0\.06838376113331358|0\.039345835316920576|1\.0|-0\.18292874508943438|0\.18232907088495806|
|PARAMETER\_8|0\.003442055686773596|0\.05645086639110787|-0\.1550985266629161|-0\.04533840533278488|0\.06898399553363642|0\.0842034527574188|-0\.14520089333558753|-0\.18292874508943438|1\.0|-0\.22415365418284647|
|PARAMETER\_9|0\.010322844346048021|0\.004495343750986927|-0\.025561490682054946|0\.0017472577315428624|0\.026635553646687377|0\.06667492174678759|0\.2700329859552825|0\.18232907088495806|-0\.22415365418284647|1\.0

## Model 
Task now is to build an effective multi-layered perceptron classifier that will be capable of telling fraud transactions with accuracy.

```python
model = Sequential([               
        tf.keras.Input(shape=(9,)),    
        Dense(128,activation ='relu'),
        Dense(64,activation ='relu'),
        Dense(32,activation ='relu'),
        Dense(1,activation ='sigmoid')
    ])
```

I have modeled the neural nets using the previous knowledge of similar dataset with binary outputs.

I found the best architecture for the Neural Network by trial-and-error (not shown here). I observed that having 3 layers improved both the training and validation loss as compared to 2 layers, and adding an extra 4th layer worsened the validation loss due to overfitting of data. The number of neurons in each layer was determined by trying several variations. On making 256 neurons in one of the layers,the estimate was satisfactory.

I experimented with various layers for the model, the reason of choosing relu was simply cause it gave the best fit, last layer is sigmoid because it works better in binary cases.


After trying several learning rates I settled on 0.001 because decreasing it further was significantly increasing the training time without much improvement on the results.

## DATA SPLIT
For splitting the data I have used stratified sampling,The purpose of stratified sampling is to ensure that each stratum is well-represented in the data, and that the sample accurately reflects the distribution of the characteristic being studied in the data.
```python
from sklearn.model_selection import StratifiedShuffleSplit
n_splits = 5
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
for train_index, test_index in sss.split(data, y_array):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]
```
Final implementation-

```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    matrics = ['accuracy']
)

model.fit(
    X_train,y_train,
    batch_size=90,
    epochs=30
)
```
As mentioned in the task, we had to expirement with loss functions etc. So here I have used BinaryCrossentropy as the loss function.
The ideal batch_size and epochs are all expiremental found by hit and trial.




## Performance
These were the final loss and accuracy on the test data that was randomly generated.

Test loss: 0.20791491866111755
Test accuracy: 0.939949095249176

AUC-ROC:  0.5086431629750349
F1 Score:  0.9690071791279987
Were also quite manageable 



## Thanks:)
