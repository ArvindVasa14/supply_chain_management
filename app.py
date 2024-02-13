from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load preprocessor and model
with open('artifacts/preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('artifacts/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html', message='Supply Chain Management')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        columns= ['Location_type', 'WH_capacity_size', 'zone', 'WH_regional_zone',
       'num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
       'retail_shop_num', 'wh_owner_type', 'distributor_num', 'flood_impacted',
       'flood_proof', 'electric_supply', 'dist_from_hub', 'workers_num',
       'storage_issue_reported_l3m', 'temp_reg_mach',
       'approved_wh_govt_certificate', 'wh_breakdown_l3m', 'govt_check_l3m']
        # [['Rural', 'Small', 'South', 'Zone 6', 3, 2.0, 3, 4733.0, 'Rented',
        # 41, 0, 0, 0, 175, 27.0, 20, 0, 'A', 4, 2]]
        Location_type= (request.form.get('Location_type')) 
        WH_capacity_size= (request.form.get('WH_capacity_size')) 
        zone= (request.form.get('zone')) 
        WH_regional_zone= (request.form.get('WH_regional_zone')) 
        num_refill_req_l3m= float(request.form.get('num_refill_req_l3m')) 
        transport_issue_l1y= float(request.form.get('transport_issue_l1y')) 
        Competitor_in_mkt= float(request.form.get('Competitor_in_mkt')) 
        retail_shop_num= float(request.form.get('retail_shop_num')) 
        wh_owner_type= (request.form.get('wh_owner_type')) 
        distributor_num= float(request.form.get('distributor_num')) 
        flood_impacted= float(request.form.get('flood_impacted')) 
        flood_proof= float(request.form.get('flood_proof')) 
        electric_supply= float(request.form.get('electric_supply')) 
        dist_from_hub= float(request.form.get('dist_from_hub')) 
        workers_num= float(request.form.get('workers_num')) 
        storage_issue_reported_l3m= float(request.form.get('storage_issue_reported_l3m')) 
        temp_reg_mach= float(request.form.get('temp_reg_mach')) 
        approved_wh_govt_certificate= (request.form.get('approved_wh_govt_certificate')) 
        wh_breakdown_l3m= float(request.form.get('wh_breakdown_l3m')) 
        govt_check_l3m= float(request.form.get('govt_check_l3m')) 

        input_data= [Location_type,WH_capacity_size,zone,WH_regional_zone,num_refill_req_l3m,transport_issue_l1y,Competitor_in_mkt,retail_shop_num,wh_owner_type,distributor_num,flood_impacted,flood_proof,electric_supply,dist_from_hub,workers_num,storage_issue_reported_l3m,temp_reg_mach,approved_wh_govt_certificate,wh_breakdown_l3m,govt_check_l3m]
        print([input_data])
        input_df= pd.DataFrame([input_data], columns= columns)
        preprocessed_data = preprocessor.transform(input_df)

        # Make predictions using the model
        predictions = model.predict(preprocessed_data)

        # You can format the predictions as needed
        result = {'predictions': predictions.tolist()[0]}

        return render_template('result.html', result= result['predictions'])

        
    except Exception as e:
        raise e




@app.route('/predict2', methods=['POST'])
def predict2():
    try:
        # Get input data from the request
        input_data = request.get_json()

        columns= ['Location_type', 'WH_capacity_size', 'zone', 'WH_regional_zone',
       'num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
       'retail_shop_num', 'wh_owner_type', 'distributor_num', 'flood_impacted',
       'flood_proof', 'electric_supply', 'dist_from_hub', 'workers_num',
       'storage_issue_reported_l3m', 'temp_reg_mach',
       'approved_wh_govt_certificate', 'wh_breakdown_l3m', 'govt_check_l3m']

        # Convert JSON to DataFrame using preprocessor
        input_df = pd.DataFrame([input_data], columns= columns)
        preprocessed_data = preprocessor.transform(input_df)

        # Make predictions using the model
        predictions = model.predict(preprocessed_data)

        # You can format the predictions as needed
        result = {'predictions': predictions.tolist()}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
