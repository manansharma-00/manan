from flask import Flask, request, jsonify
import sih_02 as sname


app = Flask(__name__)


@app.route('/api/echo', methods=['GET','POST'])
def echo():
    
    try:
        # Get the 'query' parameter from the request
        data = request.get_json()

        if 'query' in data:
            input_text = data['query']

            # Create a JSON response
            ans=sname.flask_final_func(input_text)
            response = {'result': ans}
            return jsonify(response)
        else:
            # If 'query' parameter is not provided, return an error response
            return jsonify({'error': 'Query parameter is missing'}), 400

    except Exception as e:
        # Handle exceptions if any
        return jsonify({'error': str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True)
