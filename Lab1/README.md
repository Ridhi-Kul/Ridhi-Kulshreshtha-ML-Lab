# Machine Learning Lab: Housing Price Data Analysis
### Project Overview
This lab focuses on exploratory data analysis (EDA) and statistical summary of the USA Housing dataset. The goal is to understand the features that influence housing prices through data cleaning, descriptive statistics, and visualization.

### Dataset
The analysis uses the USA_Housing.csv dataset, which contains the following features:

Avg. Area Income: Average income of residents in the city.

Avg. Area House Age: Average age of houses in the same city.

Avg. Area Number of Rooms: Average number of rooms for houses in the same city.

Avg. Area Number of Bedrooms: Average number of bedrooms for houses in the same city.

Area Population: Population of the city.

Price: Price that the house sold for (Target Variable).

Address: Address of the house.

### Implementation Steps
1. Data Loading and Preprocessing
Imported essential libraries: pandas, numpy, matplotlib, and seaborn.

Loaded the dataset and performed initial inspections using .head(), .info(), and .describe().

Verified data types and checked for non-null counts to ensure data integrity.

2. Statistical Analysis
Generated statistical summaries including mean, standard deviation, and percentiles (10%, 25%, 50%, 75%, 90%).

Identified column headers to categorize features for plotting.

3. Data Visualization
Implemented Pair Plots using Seaborn to visualize the relationships between all numerical variables.

Analyzed the distributions and scatter patterns to identify potential linear relationships with the house price.

4. Correlation Analysis
Calculated the correlation matrix for numerical features.

Identified that Avg. Area Income has the strongest positive correlation (~0.64) with the house price.