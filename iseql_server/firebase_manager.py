from typing import (
    Dict,
    Tuple,
    List,
)

import threading
import time
import os
import sys

import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class FirebaseManager(object):
    def __init__(self):
        # Use the application default credentials
        cred = credentials.Certificate(os.path.join(
            os.path.dirname(__file__),
            '..',
            'private',
            'firebase_credentials.json',
        ))
        self.default_app = firebase_admin.initialize_app(cred)
        firebase_admin.initialize_app(cred, {
            'projectId': 'active-learning-sequences',
        }, name='al_seq_firebase_manager')
        self.db = firestore.client()
    
    def save_survey_data(self, data: Dict[str, object]) -> bool:
        doc_ref = self.db.collection(u'turk_metrics').document(data['user_name'])
        doc_ref.set(data)
    
    def user_exists(self, user_name):
        res = self.db.collection(u'turk_metrics').document(user_name).get().to_dict()
        return res is not None and len(res) > 0
    
    def refresh(self):
        self.user = self.auth.refresh(self.user['refreshToken'])

class FirebaseManagerThread(threading.Thread):
    def set_firebase_manager(self, fire_base_manager: FirebaseManager):
        self.fire_base_manager = fire_base_manager

    def run(self):
        while True:
            self.fire_base_manager.refresh()
            
            # run this every 30 minutes
            time.sleep(60 * 30)

def setup_firebase() -> Tuple[FirebaseManager, threading.Thread]:
    fm = FirebaseManager()
    return fm, None