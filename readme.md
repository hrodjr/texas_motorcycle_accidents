# Texas Motorcycle Accidents

- Purpose: Analyze Texas motorcycle accidents between 2011 and 2021.
- Goal: Determine high accident areas throughout the state that ended in fatalities. 

# Project Details

- Libraries
    - pandas
    - numpy
    - seaborn
    - statsmodels
    - scipy
    - matplotlib
    - math
    - datetime
    - sklearn
- Project Modules
    - explore
    - wrangle
    
# Pipeline
My methodology follow is the data pipeline; plan, acquire, prepare, explore, model and deliver.

## Plan
- Acquire and prepare Texas motorcycle fatalities.
- Analysis based on Classification modeling and ran through the data pipeline.
- A copy of the cleaned dataset is available below.
## Acquire
- Data acquired from Texas Crash Information Record System <a href="https://cris.dot.state.tx.us/public/Query/app/home">(C.R.I.S Query)</a>.
## Prepare
- Down to 8293 rows and 11 columns.
- Encoded male, helmet, driver, deceased and injured columns.
- Dropped 726 latitude rows that had no values.
- Concated date and time, converted to datetime and set it as the index.
- Filled age 'No Data' with average age.
- Data types converted as necessary.
- A lot of background data prep was done an annotated in the 'cleaned' function in the wrangle.py.
## Explore
- Tableau
- Univariate
- Bivariate
- Multivariate
## Model
- Decision Tree
- Random Forrest
- KNN
- LogRegression

# Hypothesis
1. How likely is a rider to survive a motorcycle accident not wearing a helmet? (chi2)
    - We reject the null and accept the alternate: It is likely the rider will survive a motorcycle accident if not wearing a helmet.
2. Are people over 40 more likely to die in a motorcycle accident? (ttest 1samp)
    - We reject the null and accept the alternate: People over 40 are more likely to die in a motorcycle accident.
3. Rider with a passenger who is more likely to die in a motorcycle accident; passenger or driver?(ttest 2samp)
    - We reject the null and accept the alternate: Drivers are more likely to die in a motorcycle accident.
    
# Hypothesis Takeaways
1. Riders not wearing a helmet are likely to survive a motorcycle accident.
    - This tells me motorcycle accidents may not be severe enough to cause a fatal casualty. Good news, I guess, for riders - atleast for those in Texas!
3. Riders older than 40 are more likely to die in a motorcycle accident. 
    - Intersting my initial thoughts would be younger riders would be invovlved in more fatal causualties with their inexperience.
    - However; for older riders, this could mean several things: too comfortable, over confident, less reactive, and no recent safety training/riderskills.
4. Drivers are more then likely to die in when riding with a passenger.
    - Makes me wonder if the drive's death is compounded by the force of the passenger.
    - On the other hand is the driver acting as a buffer for the passenger and not taking the full brunt of the accident.
    - What I don't know is if the motorcycle was hit head on, by either side or rear ended.
    
# Key Takeaways and Findings
- 8293 rows and 11 columns.
- Average age of riders involved in an accident are 39 years old.
- Majority of riders who died are in the age range of 40 - 50 years old.
- Riders not wearing a helmet are likely to survive a motorcycle accident.
- Seems more older females die in accidents over 40. Where males make up a larger portion of all rider ages.
- Although the train dataset did better then the baseling in the Rnadon Forrest model both validate and test datasets did not do so well.

### Tableau
- Motorcycle Deaths in Texas between 2011-2021.
- Highest fatality areas:
    1. Dallas 
    2. Houston
    3. San Antonio
    4. Austin
    5. El Paso
- Deaths by Age: Average age is 38. Highest numbers of fatalities by age is 58.
- Death by Hour: Highest death rates at 0200 and 2100 hours.

### Acquire
- Began with 9055 columns and 39 rows.
- Convert datatypes as needed.
- Combine date and time columns. Set date index.
- Drop unnecessary columns.
- Rename columns.
- Encode columns.
- Although there are no nulls I did have to clean up 'UNKNOWN and No Data' values.

### Prepare
- Down to 8293 rows and 11 columns.
- Encoded male, helmet, driver, deceased and injured columns.
- Dropped 726 latitude rows that had no values.
- Concated date and time, converted to datetime and set it as the index.
- Filled age 'No Data' with average age.
- Data types converted as necessary.
- A lot of background data prep was done an annotated in the 'cleaned' function in the wrangle.py.

## Explore

### Univariate
1. 81.2% riders involved in a motorcycle accident are injured.
2. 92.5% riders involved in a motorcycle accident are the drivers.
3. About half of riders involved in a motorcylce accident where a helmet.
4. 88% are male riders.
5. Average age of riders involved in an accident are 39 years old.

### Bivariate
1. Of those riders injured in the accident 58 died (train dataset).
2. Drivers account for 169 deaths and passangers account for 23.
3. 118 riders who died 118 were NOT wearing a helmet.
4. Riders who died are almost equal in gender.
5. Majority of riders who died are in the age range of 40 - 50 years old.

### Multivariate
1. Violin plot confirms age is a factor in those riders who get in more accidents that conclude in injuy or death.
2. Although you are likely to survive when not wearing a helmet the violin plot show riders 40 yrs. and older who do not wear a helment result in a fatal casualty.
3. Seems more older females die in accidents over 40. Where males make up a larger portion of all rider ages.
4. Boxplot shows a slightly higher fatality average than injuries.

# Data Dictionary
| **Value** | **Descritption**                                                 | **Dtype** |
|-----------|------------------------------------------------------------------|-----------|
| city      | Texas city the accident occured.                                 | Object    |
| county    | Texas county accident occurred.                                  | Object    |
| deceased  | Death caused by the accident.                                    | int64     |
| injured   | Injury caused by the accident.                                   | int64     |
| day       | Day of the week the accident occurred.                           | Object    |
| latitude  | Location the accident occurred.                                  | float64   |
| longitude | lLocation the accident occurred.                                 | float64   |
| age       | Age of the person injured.                                       | int64     |
| driver    | Whether the person injured was the driver of the vehicle or not. | int64     |
| helmet    | Whether or not the person injured was wearing a helmet.          | int64     |
| male      | Whether or not the person injured was a male.                    | int64     |

