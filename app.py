from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Directory containing CSV files
CSV_DIR = 'csv_files/'


routes = {}

def create_routes():
    for csv_file in os.listdir(CSV_DIR):
        if csv_file.endswith('.csv'):
            # Remove file extension for endpoint name
            endpoint_name = os.path.splitext(csv_file)[0]

            # Define a unique route function for each endpoint
            def route_function(filename=csv_file):
                def inner_function():
                    df = pd.read_csv(os.path.join(CSV_DIR, filename))
                    return jsonify(df.to_dict(orient='records'))
                return inner_function

            # Map the route function to the endpoint
            routes[endpoint_name] = route_function()

            # Register the route
            app.add_url_rule(f'/{endpoint_name}', endpoint_name, routes[endpoint_name])
            print("hi")

# Create routes
create_routes()

if __name__ == '__main__':
    app.run(debug=True)
