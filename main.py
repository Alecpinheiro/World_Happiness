from flask import Flask, render_template, request
from calculate import get_dystopic_country_prediction

# Create the Flask application
app = Flask(__name__)


def cleanup(value):
    return float(value) if value else 0.0


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form inputs and clean up the values
        gdp = cleanup(request.form['gdp'])
        social_support = cleanup(request.form['socialSupport'])
        life_expectancy = cleanup(request.form['lifeExpectancy'])
        freedom = cleanup(request.form['freedom'])
        generosity = cleanup(request.form['generosity'])
        corruption = cleanup(request.form['corruption'])
        population = cleanup(request.form['population'])
        growth_rate = cleanup(request.form['growthRate'])
        power_distance = cleanup(request.form['powerDistance'])
        individualism = cleanup(request.form['individualism'])
        masculinity = cleanup(request.form['masculinity'])
        uncertainty = cleanup(request.form['uncertainty'])
        long_term = cleanup(request.form['longTerm'])
        indulgence = cleanup(request.form['indulgence'])
        regional_indicator = request.form['regionalIndicator']

        # Calculate the Happiness Score
        happiness_score = get_dystopic_country_prediction(gdp, social_support, life_expectancy, freedom, generosity, corruption, population, growth_rate, power_distance, individualism, masculinity, uncertainty, long_term, indulgence, regional_indicator);

        # Display the result
        return render_template('dystopian_country.html', happiness_score=happiness_score)

    # Render the quiz form
    return render_template('dystopian_country.html')


if __name__ == '__main__':
    app.run(debug=True)
