from flask import Flask, request, jsonify
from flask_cors import CORS

import utils
import inv1_eval, inv3_eval

app = Flask(__name__)

CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

"""
{
    "random": true,
    "speckRounds": 5
}
"""
@app.route('/sample')
def generate_sample():
    data = request.get_json()

"""
{
    "numCiphertexts": 4,
    "ciphertexts": ["0x00400000", "0x00400000", "0x00400000", "0x00400000"],
    "numRounds": 5
}
"""
@app.route('/demo', methods=['POST'])
def demo():
    data = request.get_json()
    numCiphertexts = data['numCiphertexts']
    nr = data['numRounds']
    ciphertexts = []
    for c in data['ciphertexts']:
        ciphertexts.append(utils.parse_hex_string(c))
    prob = None
    pred = None
    if numCiphertexts == 4:
        prob, pred = inv1_eval.demo_func(nr, ciphertexts)
    elif numCiphertexts == 8:
        prob, pred = inv3_eval.demo_func(nr, ciphertexts)
    res = {
        'numCiphertexts': numCiphertexts,
        'prob': str(prob),
        'pred': bool(pred)
    }
    print('res:', res)
    return jsonify(res)

@app.route('/eval', methods=['POST'])
def eval():
    data = request.get_json()
    inv = data['inv']
    eval_result = None
    if inv == 'inv1':
        eval_result = inv1_eval.eval_func()
    elif inv == 'inv3':
        eval_result = inv3_eval.eval_func()
    print('eval_result:', eval_result)
    return jsonify(eval_result)
