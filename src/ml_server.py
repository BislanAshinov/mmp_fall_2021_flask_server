from flask import Flask, render_template, redirect, request
from models import RandomForestMSE, GradientBoostingMSE
import inspect

app = Flask(__name__, template_folder='html')

rf_params = inspect.getfullargspec(RandomForestMSE.__init__).args[1:]
gb_params = inspect.getfullargspec(GradientBoostingMSE.__init__).args[1:]
rf_params.remove('estimator')
gb_params.remove('estimator')
current_model = 0
current_model_params = []
values = {}


@app.route('/')
def begin():
    global current_model
    current_model_params.clear()
    current_model = 0
    values.clear()
    return render_template("index.html")


@app.route('/hello_world')
def get_index():
    return '<html><center><script>document.write("HeLlO, wOrLd!")</script></center></html>'


# @app.route('/messages', methods=['GET', 'POST'])
# def prepare_message():
#     return redirect('/')


@app.route('/random_forest')
def choose_random_forest():
    global current_model, current_model_params
    current_model = RandomForestMSE
    current_model_params = rf_params
    return render_template('input_params.html', modelname='Random Forest', params=rf_params)


@app.route('/gradient_boosting')
def choose_gradient_boosting():
    global current_model, current_model_params
    current_model = GradientBoostingMSE
    current_model_params = gb_params
    return render_template('input_params.html', modelname='Gradient Boosting', params=gb_params)


@app.route('/init_info', methods=['GET', 'POST'])
def init_info():
    global current_model, current_model_params, values
    if request.method == 'POST':
        values = {}
        for param in current_model_params:
            try:
                values[param] = float(request.form[param])
            except ValueError:
                return render_template('error.html', error_message=f"incorrect input for parameter {param}\n")

        current_model = current_model(**values)

