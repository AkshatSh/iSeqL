import os
import sys
import syslog
import json
import pickle
import torch.multiprocessing as mp
import logging
import logging.handlers
import time

import flask
from flask import (
    Flask,
    jsonify,
    request,
    abort,
    Blueprint,
)
from flask_restful import Api
from flask_cors import CORS
import torch

import ner
import database_manager
import active_learning_manager
import survey_utils
from thread_manager import start_thread_manager, ThreadManager
from firebase_manager import setup_firebase
from server_args import get_args
from server_manager import (
    ServerManager,
)

thread_manager, thread_manager_thread = start_thread_manager()
firebase_manager, fb_manager_thread = setup_firebase()

SERVER_STATE = {
    "session_id": None,
    "ner_class": None,
    "database_manager": database_manager.DatabaseManager(),
    "al_manager": None,
    "thread_manager": thread_manager,
    "thread_manager_thread": thread_manager_thread,
    "firebase_manager": firebase_manager,
    "fb_manager_thread": fb_manager_thread,
}

def get_key_from_data(data, key):
    if key in data:
        return data[key]
    return None

def get_keys_from_data(data, *args):
    return [get_key_from_data(data, key) for key in args]

def create_app(config_filename: str):
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)
    app = Flask(__name__)
    CORS(app, resources=r'/api/*')

    # app.config.from_object(config_filename)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

def al_prepare(
    ner_class: str,
    db_manager: database_manager.DatabaseManager,
    thread_manager: ThreadManager,
    device: str,
) -> active_learning_manager.ActiveLearningManager:
    al_manager = active_learning_manager.ActiveLearningManager(
        db=db_manager,
        thread_manager=thread_manager,
        ner_class=ner_class,
        session_dir=db_manager.session_dir,
        device=device,
    )
    return al_manager

def setup_logger():
    handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", "logs/server_logs.log")
    )
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)

def setup_app(app: Flask, args: object):
    setup_logger()

    # if enabled allow the GPU to run the server, else use the CPU
    torch_device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {torch_device}...')
    logging.info(f'Using device: {torch_device}...')
    SERVER_STATE["server_manager"] = ServerManager(thread_manager, torch_device)
    if args.load_state:
        with open("server_state.pkl", 'rb') as f:
            state = pickle.load(f)
            SERVER_STATE["server_manager"].load_state(state, "server_manager")

    @app.route('/api/set_session/', methods=['GET'])
    def create_experiment():
        session_id, ner_class, user_name = get_keys_from_data(request.args, 'session_id', 'ner_class', 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        user_state = server_manager.get_user_state(user_name)
        building = user_state.set_session_state(session_id=session_id, ner_class=ner_class)
        if not building:
            logging.info("Preserving state, no need to reload")
        return jsonify({
            "messsage": "success"
        }), 201
    
    @app.route('/api/get_query/', methods=['GET'])
    def get_query():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        user_state = server_manager.get_user_state(user_name)
        al_manager = user_state.get_al_manager()
        if al_manager.get_is_start():
            user_state.set_start_time()

        query, query_predictions, final_batch = al_manager.get_query()
        if query is None and user_state.get_end_time() is None:
            user_state.set_end_time()
        return jsonify({
            "results": query,
            "predictions": query_predictions,
            "final_batch": final_batch,
        }), 201
    
    @app.route('/api/add_examples/', methods=['POST'])
    def add_examples():
        '''
        assuming arguments come as a list of (sentence_id, sentence, label)
        in an arugment called examples
        '''
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()

        json_data = json.loads(request.data)
        examples = get_keys_from_data(json_data, 'data')[0]
        sentence_ids = []
        labels = []
        try:
            for s_id, (s, ranges) in examples.items():
                sentence_ids.append(int(s_id))
                labels.append(al_manager.convert_word_ranges_to_labels(s, ranges))
            al_manager.update_examples(sentence_ids, labels)
            return  jsonify({
                "messsage": "success"
            }), 201
        except:
            with open('log.txt', 'w') as f:
                f.writelines(str(json_data))
                f.writelines(str(examples.items()))

    @app.route('/api/train/', methods=['GET'])
    def get_experiment():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        thread_id = al_manager.train()
        return jsonify({
            'thread_id': thread_id,
        }), 200
    
    @app.route('/api/evaluate/', methods=['GET'])
    def get_all_experiments():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        eval_results = al_manager.evaluate()
        return jsonify(
            eval_results
        ), 200
    
    @app.route('/api/sessions/', methods=['GET'])
    def get_all_sessions():
        db_manager = SERVER_STATE["database_manager"]
        return jsonify(
            db_manager.get_all_session_info()
        ), 200
    
    @app.route('/api/predictions/', methods=['GET'])
    def get_predictions():
        # retrieves precomputed predictions
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        if al_manager is None:
            return jsonify(
                {"error": "unknown user"}
            ), 200

        return jsonify(
            al_manager.get_predictions()
        ), 200
    
    @app.route('/api/progress/', methods=['GET'])
    def get_progress():
        thread_id = int(get_keys_from_data(request.args, 'thread_id')[0])
        thread_manager = SERVER_STATE["thread_manager"]
        prog = thread_manager.get_progress(thread_id)
        return jsonify(
            prog
        ), 200
    
    @app.route('/api/trainer_progress/', methods=['GET'])
    def get_trainer_progress():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        res = al_manager.get_training_progress()
        return jsonify(
            res,
        ), 200
    
    @app.route('/api/compute_ground_truth/', methods=['GET'])
    def get_ground_truth_result():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        res = al_manager.evaluate_ground_truth()
        return jsonify(
            res,
        ), 200
    
    @app.route('/api/users/', methods=['GET'])
    def get_all_users():
        server_manager = SERVER_STATE["server_manager"]
        return jsonify(
            server_manager.get_all_users()
        ), 200
    
    @app.route('/api/add_users/', methods=['POST'])
    def add_user():
        server_manager = SERVER_STATE["server_manager"]

        json_data = json.loads(request.data)
        user_name = get_key_from_data(json_data, 'user_name')
        res = server_manager.add_user(user_name)
        return jsonify(
            res
        ), 201
    
    @app.route('/api/save_server_state/', methods=['GET'])
    def save_server_state():
        state = {}
        server_manager = SERVER_STATE["server_manager"]
        server_manager.save_state(state, "server_manager")
        with open("server_state.pkl", "wb") as f:
            pickle.dump(state, f)

        return jsonify(
            "Success"
        ), 201
    
    @app.route('/api/save_model/', methods=['GET'])
    def save_user_model():
        server_manager = SERVER_STATE["server_manager"]
        user_name = get_key_from_data(request.args, 'user_name')
        al_manager = server_manager.get_user_state(user_name).save_model()

        return jsonify(
            "Success"
        ), 201
    
    @app.route('/api/load_model/', methods=['GET'])
    def load_user_model():
        server_manager = SERVER_STATE["server_manager"]
        user_name = get_key_from_data(request.args, 'user_name')
        al_manager = server_manager.get_user_state(user_name).load_model()

        return jsonify(
            "Success"
        ), 201
    
    @app.route('/api/submit_turk_survey/', methods=['POST'])
    def submit_turk_survey():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        firebase_manager = SERVER_STATE["firebase_manager"]
        server_manager = SERVER_STATE["server_manager"]
        user_state = server_manager.get_user_state(user_name)
        al_manager = server_manager.get_user_state(user_name).get_al_manager()
        preds = al_manager.get_predictions() 

        json_data = json.loads(request.data)
        survey_info = json_data['submit_data']
        survey_info['user_name'] = user_name
        survey_info['start_time'] = user_state.get_start_time()
        survey_info['end_time'] = user_state.get_end_time()
        survey_info['labeled_example_sizes'] = preds['labeled_set_sizes']
        survey_info['training_summary'] = preds['training_summary']
        survey_info['flipped_data'] = preds['flipped_data']
        survey_info['experiment_stats'] = al_manager.experiment_stats

        survey_code = survey_utils.submit_survey(firebase_manager, survey_info)

        try:
            server_manager.get_user_state(user_name).save_model()
            server_manager.get_user_state(user_name).save_data()
        except Exception as e:
            print('Error ocurred while attempting to save state')
            logging.exception(e)

        return jsonify(
            {"survey_code": survey_code}
        ), 201
    
    @app.route('/api/has_completed_task/', methods=['GET'])
    def check_if_user_completed_task():
        server_manager = SERVER_STATE["server_manager"]
        user_name = get_key_from_data(request.args, 'user_name')
        firebase_manager = SERVER_STATE["firebase_manager"]
        exists = firebase_manager.user_exists(user_name)

        return jsonify(
            {"exists": exists}
        ), 200
    

    @app.route('/api/used_cheatsheet/', methods=['GET'])
    def used_cheatsheet():
        user_name = get_key_from_data(request.args, 'user_name')
        server_manager = SERVER_STATE["server_manager"]
        user_state = server_manager.get_user_state(user_name)
        al_manager = user_state.get_al_manager()
        al_manager.experiment_stats["used_cheatsheet"].append(time.time())
        return jsonify(
            {"success": True}
        ), 200


def main():
    args = get_args().parse_args() 

    # set up multiprocessing
    mp.set_start_method('spawn', force=True)
    torch.set_num_threads(args.num_cpu_threads)
    os.environ["OMP_NUM_THREADS"] = f'{args.num_cpu_threads}'
    os.environ["MKL_NUM_THREADS"] = f'{args.num_cpu_threads}'

    app = create_app("config")
    setup_app(app, args)
    try:
        app.run(
            host='0.0.0.0',
            debug=args.debug,
            port=args.port,
        )
    except KeyboardInterrupt:
        print('Recieved Keyboard interrupt, saving and exiting')

if __name__ == "__main__":
    main()