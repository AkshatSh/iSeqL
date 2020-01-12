import os
import sys

from typing import (
    List,
    Dict,
    Tuple,
)

import time
import database_manager
import active_learning_manager
from thread_manager import ThreadManager

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


class ServerManagerUserNotFoundException(Exception):
    pass

class UserStateInterface(object):
    def get_al_manager(self) -> active_learning_manager.ActiveLearningManager:
        pass
    
    def get_ner_class(self) -> str:
        pass
    
    def get_session_id(self) -> int:
        pass
    
    def get_database_manager(self) -> database_manager.DatabaseManager:
        pass
    
    def set_ner_class(self, ner_class: str):
        pass
    
    def set_session_id(self, session_id: int):
        pass
    
    def set_start_time(self):
        pass
    
    def set_end_time(self):
        pass
    
    def get_start_time(self) -> float:
        pass
    
    def get_end_time(self) -> float:
        pass

    def set_al_manager(self, al_manager: active_learning_manager.ActiveLearningManager):
        pass
    
    def set_database_manager(self, db_manager: database_manager.DatabaseManager):
        pass
    
    def set_session_state(
        self,
        session_id: int,
        ner_class: str,
    ) -> bool:
        pass

class UserState(UserStateInterface):
    def __init__(self, user_name: str,  thread_manager: object, device: str):
        self.user_name = user_name
        self.session_id = None
        self.ner_class = None
        self.database_manager = database_manager.DatabaseManager()
        self.al_manager = None
        self.device = device
        self.thread_manager = thread_manager
        self.start_time = None
        self.end_time = None
    
    def set_start_time(self) -> UserStateInterface:
        self.start_time = time.time()
        return self
    
    def set_end_time(self) -> UserStateInterface:
        self.end_time = time.time()
        return self
    
    def get_start_time(self) -> float:
        return self.start_time
    
    def get_end_time(self) -> float:
        return self.end_time
    
    def get_al_manager(self) -> active_learning_manager.ActiveLearningManager:
        return self.al_manager
    
    def get_ner_class(self) -> str:
        return self.ner_class
    
    def get_session_id(self) -> int:
        return self.session_id
    
    def get_database_manager(self) -> database_manager.DatabaseManager:
        return self.database_manager
    
    def set_ner_class(self, ner_class: str) -> UserStateInterface:
        self.ner_class = ner_class
        return self
    
    def set_session_id(self, session_id: int) -> UserStateInterface:
        self.session_id = session_id
        return self
    
    def set_al_manager(self, al_manager: active_learning_manager.ActiveLearningManager) -> UserStateInterface:
        self.al_manager = al_manager
        return self
    
    def set_database_manager(self, db_manager: database_manager.DatabaseManager) -> UserStateInterface:
        self.database_manager = db_manager
        return self
    
    def set_session_state(
        self,
        session_id: int,
        ner_class: str,
    ) -> bool:
        if self.session_id != session_id or ner_class != self.ner_class:
            self.database_manager.set_session(session_id)
            self.database_manager.prepare()

            al_manager = al_prepare(
                ner_class,
                self.database_manager,
                self.thread_manager,
                self.device,
            )

            self.set_session_id(session_id).set_ner_class(ner_class).set_al_manager(al_manager)
            return True
        else:
            return False
    
    def save_model(self):
        save_dir = os.path.join(self.database_manager.session_dir, self.user_name+"/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.get_al_manager().save_model(os.path.join(save_dir, "model.ckpt"))
    
    def save_data(self):
        save_dir = os.path.join(self.database_manager.session_dir, self.user_name+"/")
        self.get_al_manager().conll_data(save_dir)

    def load_model(self):
        save_dir = os.path.join(self.database_manager.session_dir, self.user_name+"/")
        self.get_al_manager().load_model(os.path.join(save_dir, "model.ckpt"))

    def save_state(self, state_dict: Dict[str, object], key: str):
        state_dict[key] = {
            "al_manager_state": {},
            "database_manager_state": {},
            "state": (
                self.user_name,
                self.session_id,
                self.ner_class,
                self.device,
            )
        }

        self.get_al_manager().save_state(state_dict[key], "al_manager_state")
        self.get_database_manager().save_state(state_dict[key], "database_manager")

    def load_state(self, state_dict: Dict[str, object], key: str):
        (
            self.user_name,
            self.session_id,
            self.ner_class,
            self.device,
        ) = state_dict[key]["state"]
        self.database_manager.set_session(self.session_id)

        self.database_manager.load_state(state_dict[key], "database_manager")
        self.database_manager.prepare()
        al_manager = al_prepare(
            self.ner_class,
            self.database_manager,
            self.thread_manager,
            self.device,
        )

        al_manager.load_state(state_dict[key], "al_manager_state")
        self.set_al_manager(al_manager)


class ServerManager(object):
    def __init__(self, thread_manager: ThreadManager, device: str):
        self.users: Dict[str, UserState] = {}
        self.thread_manager = thread_manager
        self.device = device
    
    def get_thread_manager(self):
        return self.thread_manager
    
    def get_all_users(self) -> List[str]:
        return list(self.users.keys())
    
    def add_user(self, user_name: str) -> UserState:
        if user_name in self.users:
            return None
        self.users[user_name] = UserState(user_name, self.thread_manager, self.device)
    
    def get_user_state(self, user_name: str) -> UserState:
        if user_name not in self.users:
            raise ServerManagerUserNotFoundException(f'No such user: {user_name}')
        return self.users[user_name]
    
    def get_thread_manager(self) -> ThreadManager:
        return self.thread_manager
    
    def save_state(self, state_dict: Dict[str, object], key: str):
        state_dict[key] = {
            "device": self.device,
            "users": {},
        }

        for user_name, state in self.users.items():
            state.save_state(state_dict[key]["users"], user_name)
    
    def load_state(self, state_dict: Dict[str, object], key: str):
        self.device = state_dict[key]["device"]
        for user_name, state in state_dict[key]["users"].items():
            ustate = UserState(user_name, self.thread_manager, self.device)
            ustate.load_state(state_dict[key]["users"], user_name)
            self.users[user_name] = ustate