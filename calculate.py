import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

this_folder = Path(__file__).parent.resolve()

def get_dystopic_country_prediction(gdp,
                                   social_support,
                                   life_expectancy,
                                   freedom,
                                   generosity,
                                   corruption,
                                   population,
                                   growth_rate,
                                   power_distance,
                                   individualism,
                                   masculinity,
                                   uncertainty,
                                   long_term,
                                   indulgence,
                                   regional_indicator):

    # Read the dataset into a DataFrame
    df_x = pd.read_csv(this_folder/'df_x.csv')
    df_target = pd.read_csv(this_folder/'df_target.csv')

    # Split the dataset into input features (X) and target variable (y)
    X = df_x
    y = df_target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

    dystopic_country = pd.DataFrame({
        'Logged GDP per capita': [gdp],
        'Social support': [social_support],
        'Healthy life expectancy': [life_expectancy],
        'Freedom to make life choices': [freedom],
        'Generosity': [generosity],
        'Perceptions of corruption': [corruption],
        'Population in 2021': [population],
        'Growth rate': [growth_rate],
        'Power distance': [power_distance],
        'Individualism': [individualism],
        'Masculinity': [masculinity],
        'Uncertainty avoidance': [uncertainty],
        'Long term orientation': [long_term],
        'Indulgence': [indulgence],
        'Regional indicator_Central and Eastern Europe': [0],
        'Regional indicator_Commonwealth of Independent States': [0],
        'Regional indicator_East Asia': [0],
        'Regional indicator_Latin America and Caribbean': [0],
        'Regional indicator_Middle East and North Africa': [0],
        'Regional indicator_North America and ANZ': [0],
        'Regional indicator_South Asia': [0],
        'Regional indicator_Southeast Asia': [0],
        'Regional indicator_Sub-Saharan Africa': [0],
        'Regional indicator_Western Europe': [0],
    })

    # Make predictions for the dystopic country
    get_dystopic_pred = model.predict(dystopic_country)
    return get_dystopic_pred[0]


